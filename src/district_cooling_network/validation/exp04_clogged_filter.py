
"""Experiment 04 — matched clean vs clogged-filter validation."""

from pathlib import Path

import numpy as np
import pandas as pd

import district_cooling_network.bootstrap  # noqa: F401
from district_cooling_network.paths import get_workspace

from dcngen.config import load_config
from dcngen.loads.archetypes import ARCHETYPE_NAMES
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import design_ua
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.topology.dcnifier import build_dcn, _size_plant
from dcngen.topology.ditec_loader import load_ditec_network


WS = get_workspace()
CFG_PATH = WS.dcngen_config()
OUT = WS.validation_dir("04_clogged_filter")
OUT.mkdir(parents=True, exist_ok=True)

# Simulation-defined sensitivity parameters.
# These are NOT literal percentages of physical clogging.
K_CLEAN = 1.0
K_CLOGGED = 20.0

# Short connector so distributed friction is negligible.
# The intended mechanism is localized minor loss K.
FILTER_LENGTH = 0.01

# Numerical thermal regularization for the artificial filter connector.
# Must be > 0 because the thermal solver computes tau ~ 1/U'.
# This is NOT a physical filter heat-transfer parameter.
FILTER_U_PRIME = 1e-6


def insert_filter(dcn, consumer_name, k_loss):
    """Insert passive localized resistance upstream of an ETS FCV."""

    wn = dcn.wn
    consumer = dcn.consumers[consumer_name]

    valve_name = consumer.ets_link
    valve = wn.get_link(valve_name)

    supply_node = valve.start_node_name
    return_node = valve.end_node_name

    diameter = float(valve.diameter)
    setting = float(valve.initial_setting)

    filter_node = f"{valve_name}_filter_node"
    filter_link = f"{valve_name}_filter"

    supply = wn.get_node(supply_node)

    wn.add_junction(
        filter_node,
        base_demand=0.0,
        elevation=float(supply.elevation),
    )

    wn.remove_link(valve_name)

    wn.add_pipe(
        filter_link,
        supply_node,
        filter_node,
        length=FILTER_LENGTH,
        diameter=diameter,
        roughness=100.0,
        minor_loss=float(k_loss),
    )

    # Register the artificial filter connector with the inherited
    # thermal topology. It lies upstream of the ETS on the supply side.
    dcn.edge_kind[filter_link] = "supply_pipe"

    wn.add_valve(
        valve_name,
        filter_node,
        return_node,
        diameter=diameter,
        valve_type="FCV",
        initial_setting=setting,
    )

    return filter_link


# ============================================================
# 1. COMMON EXOGENOUS SCENARIO
# ============================================================

cfg = load_config(CFG_PATH)

n_steps = 48
times = np.arange(n_steps) * cfg.solver.dt

master_rng = np.random.default_rng(cfg.seed)

static_seed = int(master_rng.integers(2**63))
scenario_seed = int(master_rng.integers(2**63))

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

rng = np.random.default_rng(scenario_seed)

heat_gain = resolve_pipe_heat_gain(
    template,
    cfg,
    rng,
)

design = {
    j: c.design_load
    for j, c in template.consumers.items()
}

loads = generate_loads(
    archetypes,
    design,
    times,
    cfg.loads,
    rng,
)

ua = {
    j: design_ua(design[j], cfg)
    for j in design
}


# ============================================================
# 2. SELECT TARGET
#
# First obtain the ordinary healthy trajectory so we choose
# a dynamically active terminal rather than an arbitrary ETS.
# ============================================================

reference_dcn = build_dcn(net, cfg)

print("Running reference scenario for target selection...")

reference = run_eps_ntu_scenario(
    reference_dcn,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    pipe_U=heat_gain.U,
)

desired_std = reference.desired_flow.std(axis=0)
target = str(desired_std.idxmax())

print("Selected target consumer:", target)
print(
    "Reference desired-flow std:",
    float(desired_std[target]),
)


# ============================================================
# 3. CLEAN FILTER COUNTERFACTUAL
# ============================================================

dcn_clean = build_dcn(net, cfg)

clean_filter = insert_filter(
    dcn_clean,
    target,
    K_CLEAN,
)

# Re-size the HEALTHY plant with the clean filter present.
# build_dcn() has already sized the plant once, so restore the
# inherited Big-M probe state before calling _size_plant() again.
q_design, _ = dcn_clean.pump_design

dcn_clean.wn.get_curve("plant_pump_curve").points = [
    (float(q_design), 1.0e4)
]

_size_plant(dcn_clean, cfg)

healthy_design_flow, healthy_design_head = dcn_clean.pump_design
healthy_reservoir_head = float(
    dcn_clean.wn.get_node(dcn_clean.plant.reservoir).base_head
)

print(
    "Healthy plant re-sized with clean filter:",
    f"Q_design={healthy_design_flow:.12g} m3/s,",
    f"H0={healthy_design_head:.12g} m",
)

# The inserted filter is a numerically short hydraulic connector
# representing localized resistance, not a distributed thermal pipe.
# Add it to the thermal bookkeeping with zero distributed heat gain.
pipe_U_clean = dict(heat_gain.U)
pipe_U_clean[clean_filter] = FILTER_U_PRIME

print("\nRunning CLEAN-FILTER scenario...")

clean = run_eps_ntu_scenario(
    dcn_clean,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    pipe_U=pipe_U_clean,
)

print("✓ Clean-filter complete")


# ============================================================
# 4. CLOGGED FILTER COUNTERFACTUAL
# ============================================================

dcn_clogged = build_dcn(net, cfg)

clogged_filter = insert_filter(
    dcn_clogged,
    target,
    K_CLOGGED,
)

# Freeze the plant at the CLEAN-filter healthy design.
# Do NOT re-size the plant for the clogged condition.
dcn_clogged.pump_design = (
    healthy_design_flow,
    healthy_design_head,
)

# Replace the pump HEAD curve with the healthy design point.
wn_clogged = dcn_clogged.wn
curve = wn_clogged.get_curve("plant_pump_curve")
curve.points = [
    (
        float(healthy_design_flow),
        float(healthy_design_head),
    )
]

# Preserve the healthy reservoir anchoring as well.
wn_clogged.get_node(
    dcn_clogged.plant.reservoir
).base_head = healthy_reservoir_head

# Same thermal treatment as the clean counterfactual.
pipe_U_clogged = dict(heat_gain.U)
pipe_U_clogged[clogged_filter] = FILTER_U_PRIME

print("\nRunning CLOGGED-FILTER scenario...")

clogged = run_eps_ntu_scenario(
    dcn_clogged,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    pipe_U=pipe_U_clogged,
)

print("✓ Clogged-filter complete")


# ============================================================
# 5. EXTRACT FILTER HYDRAULICS
# ============================================================

q_clean = clean.pipe_flow[clean_filter].to_numpy()
q_clogged = clogged.pipe_flow[clogged_filter].to_numpy()

v_clean = clean.pipe_velocity[clean_filter].to_numpy()
v_clogged = clogged.pipe_velocity[clogged_filter].to_numpy()

dh_clean = clean.pipe_headloss[clean_filter].to_numpy()
dh_clogged = clogged.pipe_headloss[clogged_filter].to_numpy()

consumer_clean = clean.consumer_flow[target].to_numpy()
consumer_clogged = clogged.consumer_flow[target].to_numpy()

# Local-loss theoretical component for an independent consistency check.
g = 9.80665

local_clean = K_CLEAN * v_clean**2 / (2.0 * g)
local_clogged = K_CLOGGED * v_clogged**2 / (2.0 * g)

mean_dh_clean = float(np.mean(dh_clean))
mean_dh_clogged = float(np.mean(dh_clogged))

mean_q_clean = float(np.mean(np.abs(q_clean)))
mean_q_clogged = float(np.mean(np.abs(q_clogged)))

max_flow_difference = float(
    np.max(np.abs(consumer_clogged - consumer_clean))
)

mean_local_clean = float(np.mean(local_clean))
mean_local_clogged = float(np.mean(local_clogged))

# Compare observed head-loss increase.
headloss_ratio = (
    mean_dh_clogged / mean_dh_clean
    if abs(mean_dh_clean) > 1e-12
    else np.inf
)

# Because q may change, K ratio alone is not necessarily
# the expected head-loss ratio.
expected_local_ratio = (
    mean_local_clogged / mean_local_clean
    if mean_local_clean > 1e-12
    else np.inf
)


# ============================================================
# 6. REPORT
# ============================================================

print("\n" + "=" * 72)
print("EXPERIMENT 04 — CLOGGED FILTER VALIDATION")
print("=" * 72)

print("Target consumer:", target)
print("Scenario seed:", scenario_seed)
print("Clean K:", K_CLEAN)
print("Clogged K:", K_CLOGGED)

print("\nFILTER FLOW")
print(
    "Clean mean |Q| [m3/s]:",
    f"{mean_q_clean:.12g}",
)
print(
    "Clogged mean |Q| [m3/s]:",
    f"{mean_q_clogged:.12g}",
)
print(
    "Max consumer-flow difference [m3/s]:",
    f"{max_flow_difference:.12g}",
)

print("\nFILTER HEAD LOSS")
print(
    "Clean mean Δh [m]:",
    f"{mean_dh_clean:.12g}",
)
print(
    "Clogged mean Δh [m]:",
    f"{mean_dh_clogged:.12g}",
)
print(
    "Observed mean Δh ratio:",
    f"{headloss_ratio:.12g}",
)

print("\nLOCAL-LOSS CONSISTENCY")
print(
    "Clean mean K*v²/(2g) [m]:",
    f"{mean_local_clean:.12g}",
)
print(
    "Clogged mean K*v²/(2g) [m]:",
    f"{mean_local_clogged:.12g}",
)
print(
    "Expected local-loss ratio:",
    f"{expected_local_ratio:.12g}",
)

print("\nCONVERGENCE")
print(
    "Clean:",
    int(clean.converged.sum()),
    "/",
    len(clean.converged),
)
print(
    "Clogged:",
    int(clogged.converged.sum()),
    "/",
    len(clogged.converged),
)


# ============================================================
# 7. MECHANISM CHECKS
# ============================================================

checks = {
    "filter is hydraulically active":
        mean_dh_clean > 0.0,

    "clogging increases local head loss":
        mean_dh_clogged > mean_dh_clean,

    "theoretical local loss increases":
        mean_local_clogged > mean_local_clean,

    "clean simulation converged":
        bool(np.all(clean.converged)),

    "clogged simulation converged":
        bool(np.all(clogged.converged)),
}

print("\nMECHANISM CHECKS")

for name, passed in checks.items():
    print(
        f"{'PASS' if passed else 'FAIL'}  {name}"
    )

overall = all(checks.values())

print(
    "\nOVERALL:",
    "PASS" if overall else "FAIL",
)


# ============================================================
# 8. SAVE PLOT-READY RESULTS
# ============================================================

df = pd.DataFrame(
    {
        "time_s": clean.time,
        "clean_filter_flow_m3s": q_clean,
        "clogged_filter_flow_m3s": q_clogged,
        "clean_filter_velocity_ms": v_clean,
        "clogged_filter_velocity_ms": v_clogged,
        "clean_filter_headloss_m": dh_clean,
        "clogged_filter_headloss_m": dh_clogged,
        "clean_local_loss_theory_m": local_clean,
        "clogged_local_loss_theory_m": local_clogged,
        "clean_consumer_flow_m3s": consumer_clean,
        "clogged_consumer_flow_m3s": consumer_clogged,
    }
)

out_path = (
    OUT /
    "clogged_filter_validation.csv"
)

df.to_csv(out_path, index=False)

print("\nSaved:", out_path)
