
"""
Experiment 06 — Pilot multi-FSED dataset generation.

Purpose
-------
Generate a small, traceable dataset from the five physics-validated FSED
mechanisms before scaling to the final thesis dataset.

Important:
- Severity values are simulation sensitivity levels, not universal thresholds.
- Healthy counterfactuals are retained for simulator validation/provenance.
- Counterfactual residuals are NOT automatically ML input features.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from dcngen.config import load_config
from dcngen.topology.ditec_loader import load_ditec_network
from dcngen.topology.dcnifier import build_dcn, _size_plant
from dcngen.loads.archetypes import ARCHETYPE_NAMES
from dcngen.loads.load_generator import generate_loads
from dcngen.thermal.ets import design_ua
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.faults.injector import (
    inject_poor_balancing,
    StuckValveInjection,
)
from dcngen.faults.taxonomy import FaultLabel, FaultKind


ROOT = Path("/content/drive/MyDrive/DCN_Thesis")
CFG_PATH = ROOT / "configs/colab.yaml"

OUT = ROOT / "results/06_pilot_dataset"
SCENARIO_OUT = OUT / "scenarios"

OUT.mkdir(parents=True, exist_ok=True)
SCENARIO_OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config(CFG_PATH)

N_STEPS = 48
times = np.arange(N_STEPS) * cfg.solver.dt

# ------------------------------------------------------------
# PILOT DESIGN
# ------------------------------------------------------------

N_SEEDS = 1

BALANCING_LEVELS = [0.80]
CONTROL_MARGIN_LEVELS = [1.20]
FILTER_LEVELS = [20.0]
FOULING_LEVELS = [0.85]

# Stuck actuator has no meaningful percentage severity in the current model.
# Instead vary onset time.
STUCK_ONSET_FRACTIONS = [0.50]

HEALTHY_FACTOR = 1.05

K_CLEAN = 1.0
FILTER_LENGTH = 0.01
FILTER_U_PRIME = 1e-6

PUMP_CURVE = "plant_pump_curve"

master_rng = np.random.default_rng(cfg.seed)

# Keep one deterministic static topology/archetype draw.
static_seed = int(master_rng.integers(2**63))

# Independent dynamic scenario seeds.
scenario_seeds = [
    int(master_rng.integers(2**63))
    for _ in range(N_SEEDS)
]


# ============================================================
# FILTER HELPER
# ============================================================

def insert_filter(dcn, consumer_name, k_loss):
    """
    Insert a short passive localized resistance immediately upstream
    of the target ETS FCV.

    k_loss is a simulation sensitivity parameter.
    """

    wn = dcn.wn
    consumer = dcn.consumers[consumer_name]

    valve_name = consumer.ets_link
    valve = wn.get_link(valve_name)

    supply_node = valve.start_node_name
    return_node = valve.end_node_name

    diameter = float(valve.diameter)

    filter_node = f"filter_{consumer_name}"
    filter_link = f"filter_pipe_{consumer_name}"

    if filter_node in wn.node_name_list:
        raise RuntimeError(f"Filter node already exists: {filter_node}")

    if filter_link in wn.link_name_list:
        raise RuntimeError(f"Filter link already exists: {filter_link}")

    elevation = float(wn.get_node(supply_node).elevation)

    wn.remove_link(valve_name)

    wn.add_junction(
        filter_node,
        base_demand=0.0,
        elevation=elevation,
    )

    wn.add_pipe(
        filter_link,
        supply_node,
        filter_node,
        length=FILTER_LENGTH,
        diameter=diameter,
        roughness=100.0,
        minor_loss=float(k_loss),
    )

    # Re-create the inherited ideal FCV downstream of the filter.
    wn.add_valve(
        valve_name,
        filter_node,
        return_node,
        diameter=diameter,
        valve_type="FCV",
        minor_loss=0.0,
        initial_setting=float(valve.initial_setting),
    )

    # Thermal graph bookkeeping.
    dcn.edge_kind[filter_link] = "supply_pipe"
    dcn.edge_kind[valve_name] = "ets"

    return filter_link


# ============================================================
# RESULT -> LONG TABLE
# ============================================================

def result_to_frame(
    result,
    *,
    scenario_id,
    pair_id,
    seed,
    state,
    fsed,
    severity_parameter,
    severity_value,
    target,
    onset_s=np.nan,
    filter_link=None,
    plant_head=np.nan,
):
    """
    Convert the target ETS trajectory into a compact pilot table.

    We intentionally retain raw physical channels rather than healthy-fault
    residuals as detector inputs.
    """

    target = str(target)

    q = result.consumer_flow[target].to_numpy(dtype=float)
    desired = result.desired_flow[target].to_numpy(dtype=float)

    ps = result.pressure_s[target].to_numpy(dtype=float)
    pr = result.pressure_r[target].to_numpy(dtype=float)

    # Thermal/load channels may be DataFrames indexed by consumer.
    ts = result.supply_temp[target].to_numpy(dtype=float)
    tr = result.return_temp[target].to_numpy(dtype=float)

    delivered = result.delivered_load[target].to_numpy(dtype=float)
    unmet = result.unmet[target].to_numpy(dtype=bool)

    converged = np.asarray(result.converged, dtype=bool)

    df = pd.DataFrame({
        "scenario_id": scenario_id,
        "pair_id": pair_id,
        "seed": int(seed),
        "state": state,
        "fsed": fsed,
        "severity_parameter": severity_parameter,
        "severity_value": float(severity_value),
        "target_consumer": target,
        "fault_onset_s": onset_s,
        "time_s": times.astype(float),

        "desired_flow_m3s": desired,
        "realized_flow_m3s": q,

        "supply_pressure_m": ps,
        "return_pressure_m": pr,
        "ets_dp_m": ps - pr,

        "supply_temp_C": ts,
        "return_temp_C": tr,
        "deltaT_K": np.abs(tr - ts),

        "delivered_load_W": delivered,
        "unmet": unmet,
        "converged": converged,

        "plant_head_m": plant_head,
    })

    if filter_link is not None:
        df["filter_flow_m3s"] = (
            result.pipe_flow[filter_link].to_numpy(dtype=float)
        )
        df["filter_headloss_m"] = (
            result.pipe_headloss[filter_link].to_numpy(dtype=float)
        )
    else:
        df["filter_flow_m3s"] = np.nan
        df["filter_headloss_m"] = np.nan

    return df


# ============================================================
# COMMON STATIC NETWORK
# ============================================================

net = load_ditec_network(
    Path(cfg.paths.ditec_data) / cfg.poc.network,
    cfg.poc.static_scenario_id,
)

consumers = sorted(
    j
    for j in net.junctions
    if net.draw.base_demand.get(j, 0.0) > 0.0
)

archetype_rng = np.random.default_rng(static_seed)

archetypes = {
    j: str(a)
    for j, a in zip(
        consumers,
        archetype_rng.choice(
            ARCHETYPE_NAMES,
            size=len(consumers),
        ),
    )
}

template = build_dcn(net, cfg)

design_loads = {
    j: c.design_load
    for j, c in template.consumers.items()
}


all_frames = []
manifest = []

print("=" * 78)
print("EXPERIMENT 06 — PILOT DATASET")
print("=" * 78)
print("Seeds:", len(scenario_seeds))
print("Consumers:", len(consumers))


# ============================================================
# EACH DYNAMIC SCENARIO SEED
# ============================================================

for seed_index, scenario_seed in enumerate(scenario_seeds):

    print("\n" + "=" * 78)
    print(
        f"SEED {seed_index + 1}/{len(scenario_seeds)}:",
        scenario_seed,
    )
    print("=" * 78)

    rng = np.random.default_rng(scenario_seed)

    # Same exogenous realization shared by healthy/fault twins.
    heat_gain = resolve_pipe_heat_gain(
        template,
        cfg,
        rng,
    )

    loads = generate_loads(
        archetypes,
        design_loads,
        times,
        cfg.loads,
        rng,
    )

    # Healthy UA baseline.
    ua_healthy = {
        j: design_ua(design_loads[j], cfg) * HEALTHY_FACTOR
        for j in consumers
    }

    pipe_U = dict(heat_gain.U)

    # --------------------------------------------------------
    # Reference healthy run used to select a dynamic target ETS
    # --------------------------------------------------------

    reference_dcn = build_dcn(net, cfg)

    reference = run_eps_ntu_scenario(
        reference_dcn,
        cfg,
        loads,
        n_steps=N_STEPS,
        ua=ua_healthy,
        pipe_U=pipe_U,
    )

    # Choose an ETS whose controller request actually changes.
    desired_std = {
        j: float(
            np.std(
                reference.desired_flow[j].to_numpy(dtype=float)
            )
        )
        for j in consumers
    }

    target = max(desired_std, key=desired_std.get)

    print(
        "Target:",
        target,
        "| desired-flow std:",
        desired_std[target],
    )

    # ========================================================
    # 1. STUCK ACTUATOR
    # ========================================================

    for level_i, onset_fraction in enumerate(
        STUCK_ONSET_FRACTIONS,
        start=1,
    ):
        pair_id = (
            f"seed{seed_index:02d}_stuck_{level_i:02d}"
        )

        onset_idx = int(
            round(onset_fraction * (N_STEPS - 1))
        )

        # Keep onset away from t=0.
        onset_idx = max(1, min(onset_idx, N_STEPS - 1))
        onset_s = float(times[onset_idx])

        healthy_dcn = build_dcn(net, cfg)

        healthy = run_eps_ntu_scenario(
            healthy_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )

        healthy_head = float(healthy_dcn.pump_design[1])

        h_id = pair_id + "_healthy"

        all_frames.append(
            result_to_frame(
                healthy,
                scenario_id=h_id,
                pair_id=pair_id,
                seed=scenario_seed,
                state="healthy",
                fsed="stuck_actuator",
                severity_parameter="onset_fraction",
                severity_value=float(onset_fraction),
                target=target,
                onset_s=onset_s,
                plant_head=healthy_head,
            )
        )

        fault_dcn = build_dcn(net, cfg)

        mask = times >= onset_s

        label = FaultLabel(
            kind=FaultKind.ETS_VALVE,
            location=target,
            element="node",
            side=None,
            severity=1.0,
            onset=onset_s,
            offset=None,
            mask=mask,
        )

        stuck = StuckValveInjection(
            label=label,
            consumer=target,
        )

        fault = run_eps_ntu_scenario(
            fault_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            stuck_valve=stuck,
            pipe_U=pipe_U,
        )

        f_id = pair_id + "_fault"

        all_frames.append(
            result_to_frame(
                fault,
                scenario_id=f_id,
                pair_id=pair_id,
                seed=scenario_seed,
                state="fault",
                fsed="stuck_actuator",
                severity_parameter="onset_fraction",
                severity_value=float(onset_fraction),
                target=target,
                onset_s=onset_s,
                plant_head=healthy_head,
            )
        )

        manifest.append({
            "pair_id": pair_id,
            "seed": scenario_seed,
            "fsed": "stuck_actuator",
            "severity_parameter": "onset_fraction",
            "severity_value": float(onset_fraction),
            "target_consumer": target,
            "fault_onset_s": onset_s,
        })

        print("✓", pair_id)

    # ========================================================
    # 2. POOR BALANCING
    # ========================================================

    for multiplier in BALANCING_LEVELS:

        pair_id = (
            f"seed{seed_index:02d}_balancing_{multiplier:.2f}"
        )

        healthy_dcn = build_dcn(net, cfg)

        healthy = run_eps_ntu_scenario(
            healthy_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )

        healthy_head = float(healthy_dcn.pump_design[1])

        all_frames.append(
            result_to_frame(
                healthy,
                scenario_id=pair_id + "_healthy",
                pair_id=pair_id,
                seed=scenario_seed,
                state="healthy",
                fsed="poor_balancing",
                severity_parameter="flow_multiplier",
                severity_value=float(multiplier),
                target=target,
                plant_head=healthy_head,
            )
        )

        fault_dcn = build_dcn(net, cfg)

        balancing = inject_poor_balancing(
            fault_dcn,
            cfg,
            times,
            consumer=target,
            flow_multiplier=float(multiplier),
        )

        fault = run_eps_ntu_scenario(
            fault_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            poor_balancing=balancing,
            pipe_U=pipe_U,
        )

        all_frames.append(
            result_to_frame(
                fault,
                scenario_id=pair_id + "_fault",
                pair_id=pair_id,
                seed=scenario_seed,
                state="fault",
                fsed="poor_balancing",
                severity_parameter="flow_multiplier",
                severity_value=float(multiplier),
                target=target,
                plant_head=healthy_head,
            )
        )

        manifest.append({
            "pair_id": pair_id,
            "seed": scenario_seed,
            "fsed": "poor_balancing",
            "severity_parameter": "flow_multiplier",
            "severity_value": float(multiplier),
            "target_consumer": target,
            "fault_onset_s": np.nan,
        })

        print("✓", pair_id)

    # ========================================================
    # 3. EXCESSIVE CONTROL MARGIN
    # ========================================================

    for multiplier in CONTROL_MARGIN_LEVELS:

        pair_id = (
            f"seed{seed_index:02d}_control_{multiplier:.2f}"
        )

        healthy_dcn = build_dcn(net, cfg)

        healthy = run_eps_ntu_scenario(
            healthy_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )

        q_design, h0 = healthy_dcn.pump_design

        all_frames.append(
            result_to_frame(
                healthy,
                scenario_id=pair_id + "_healthy",
                pair_id=pair_id,
                seed=scenario_seed,
                state="healthy",
                fsed="excess_control_margin",
                severity_parameter="head_multiplier",
                severity_value=float(multiplier),
                target=target,
                plant_head=float(h0),
            )
        )

        fault_dcn = build_dcn(net, cfg)

        qf, hf = fault_dcn.pump_design

        if not np.isclose(hf, h0):
            raise RuntimeError(
                "Healthy/fault base plant heads differ."
            )

        fault_head = float(h0 * multiplier)

        fault_dcn.wn.get_curve(PUMP_CURVE).points = [
            (float(qf), fault_head)
        ]

        fault = run_eps_ntu_scenario(
            fault_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )

        all_frames.append(
            result_to_frame(
                fault,
                scenario_id=pair_id + "_fault",
                pair_id=pair_id,
                seed=scenario_seed,
                state="fault",
                fsed="excess_control_margin",
                severity_parameter="head_multiplier",
                severity_value=float(multiplier),
                target=target,
                plant_head=fault_head,
            )
        )

        manifest.append({
            "pair_id": pair_id,
            "seed": scenario_seed,
            "fsed": "excess_control_margin",
            "severity_parameter": "head_multiplier",
            "severity_value": float(multiplier),
            "target_consumer": target,
            "fault_onset_s": np.nan,
        })

        print("✓", pair_id)

    # ========================================================
    # 4. CLOGGED FILTER
    # ========================================================

    for k_clogged in FILTER_LEVELS:

        pair_id = (
            f"seed{seed_index:02d}_filter_K{k_clogged:g}"
        )

        # Healthy clean-filter network.
        clean_dcn = build_dcn(net, cfg)

        clean_filter = insert_filter(
            clean_dcn,
            target,
            K_CLEAN,
        )

        # Restore inherited sizing probe before re-sizing.
        q_design, _ = clean_dcn.pump_design

        clean_dcn.wn.get_curve(PUMP_CURVE).points = [
            (float(q_design), 1.0e4)
        ]

        _size_plant(clean_dcn, cfg)

        healthy_q_design, healthy_head = (
            clean_dcn.pump_design
        )

        healthy_reservoir_head = float(
            clean_dcn.wn.get_node(
                clean_dcn.plant.reservoir
            ).base_head
        )

        clean_pipe_U = dict(pipe_U)
        clean_pipe_U[clean_filter] = FILTER_U_PRIME

        healthy = run_eps_ntu_scenario(
            clean_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=clean_pipe_U,
        )

        all_frames.append(
            result_to_frame(
                healthy,
                scenario_id=pair_id + "_healthy",
                pair_id=pair_id,
                seed=scenario_seed,
                state="healthy",
                fsed="clogged_filter",
                severity_parameter="filter_K",
                severity_value=float(k_clogged),
                target=target,
                filter_link=clean_filter,
                plant_head=float(healthy_head),
            )
        )

        # Faulted network with identical healthy plant design.
        clogged_dcn = build_dcn(net, cfg)

        clogged_filter = insert_filter(
            clogged_dcn,
            target,
            float(k_clogged),
        )

        clogged_dcn.pump_design = (
            float(healthy_q_design),
            float(healthy_head),
        )

        clogged_dcn.wn.get_curve(PUMP_CURVE).points = [
            (
                float(healthy_q_design),
                float(healthy_head),
            )
        ]

        clogged_dcn.wn.get_node(
            clogged_dcn.plant.reservoir
        ).base_head = healthy_reservoir_head

        clogged_pipe_U = dict(pipe_U)
        clogged_pipe_U[clogged_filter] = FILTER_U_PRIME

        fault = run_eps_ntu_scenario(
            clogged_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=clogged_pipe_U,
        )

        all_frames.append(
            result_to_frame(
                fault,
                scenario_id=pair_id + "_fault",
                pair_id=pair_id,
                seed=scenario_seed,
                state="fault",
                fsed="clogged_filter",
                severity_parameter="filter_K",
                severity_value=float(k_clogged),
                target=target,
                filter_link=clogged_filter,
                plant_head=float(healthy_head),
            )
        )

        manifest.append({
            "pair_id": pair_id,
            "seed": scenario_seed,
            "fsed": "clogged_filter",
            "severity_parameter": "filter_K",
            "severity_value": float(k_clogged),
            "target_consumer": target,
            "fault_onset_s": np.nan,
        })

        print("✓", pair_id)

    # ========================================================
    # 5. HX FOULING
    # ========================================================

    for multiplier in FOULING_LEVELS:

        pair_id = (
            f"seed{seed_index:02d}_fouling_{multiplier:.2f}"
        )

        healthy_dcn = build_dcn(net, cfg)

        healthy = run_eps_ntu_scenario(
            healthy_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )

        healthy_head = float(healthy_dcn.pump_design[1])

        all_frames.append(
            result_to_frame(
                healthy,
                scenario_id=pair_id + "_healthy",
                pair_id=pair_id,
                seed=scenario_seed,
                state="healthy",
                fsed="hx_fouling",
                severity_parameter="ua_multiplier",
                severity_value=float(multiplier),
                target=target,
                plant_head=healthy_head,
            )
        )

        ua_fault = dict(ua_healthy)
        ua_fault[target] = (
            ua_healthy[target] * float(multiplier)
        )

        fault_dcn = build_dcn(net, cfg)

        fault = run_eps_ntu_scenario(
            fault_dcn,
            cfg,
            loads,
            n_steps=N_STEPS,
            ua=ua_fault,
            pipe_U=pipe_U,
        )

        all_frames.append(
            result_to_frame(
                fault,
                scenario_id=pair_id + "_fault",
                pair_id=pair_id,
                seed=scenario_seed,
                state="fault",
                fsed="hx_fouling",
                severity_parameter="ua_multiplier",
                severity_value=float(multiplier),
                target=target,
                plant_head=healthy_head,
            )
        )

        manifest.append({
            "pair_id": pair_id,
            "seed": scenario_seed,
            "fsed": "hx_fouling",
            "severity_parameter": "ua_multiplier",
            "severity_value": float(multiplier),
            "target_consumer": target,
            "fault_onset_s": np.nan,
        })

        print("✓", pair_id)


# ============================================================
# SAVE PILOT DATASET
# ============================================================

dataset = pd.concat(
    all_frames,
    ignore_index=True,
)

manifest_df = pd.DataFrame(manifest)

dataset_path = OUT / "pilot_dataset.parquet"
manifest_path = OUT / "scenario_manifest.csv"
summary_path = OUT / "dataset_summary.csv"

dataset.to_parquet(
    dataset_path,
    index=False,
)

manifest_df.to_csv(
    manifest_path,
    index=False,
)

summary = (
    dataset.groupby(
        ["fsed", "state"],
        dropna=False,
    )
    .agg(
        rows=("time_s", "size"),
        scenarios=("scenario_id", "nunique"),
        seeds=("seed", "nunique"),
        convergence_rate=("converged", "mean"),
        unmet_rate=("unmet", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    summary_path,
    index=False,
)


# Dataset-level QA.
expected_pairs = (
    N_SEEDS
    * (
        len(STUCK_ONSET_FRACTIONS)
        + len(BALANCING_LEVELS)
        + len(CONTROL_MARGIN_LEVELS)
        + len(FILTER_LEVELS)
        + len(FOULING_LEVELS)
    )
)

assert manifest_df["pair_id"].nunique() == expected_pairs
assert dataset["pair_id"].nunique() == expected_pairs

pair_state_counts = (
    dataset[
        ["pair_id", "scenario_id", "state"]
    ]
    .drop_duplicates()
    .groupby("pair_id")["state"]
    .nunique()
)

assert (pair_state_counts == 2).all()

assert dataset["converged"].all(), (
    "At least one pilot timestep did not converge."
)

print("\n" + "=" * 78)
print("PILOT DATASET COMPLETE")
print("=" * 78)

print("Counterfactual pairs:", expected_pairs)
print("Physical scenarios:", dataset["scenario_id"].nunique())
print("Rows:", len(dataset))
print("Seeds:", dataset["seed"].nunique())

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(dataset_path)
print(manifest_path)
print(summary_path)
