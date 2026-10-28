
"""Experiment 05 — matched healthy vs heat-exchanger fouling validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from dcngen.config import load_config
from dcngen.loads.archetypes import ARCHETYPE_NAMES
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import installed_ua
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network


ROOT = Path("/content/drive/MyDrive/DCN_Thesis")
CFG_PATH = ROOT / "configs/colab.yaml"

OUT = ROOT / "results/05_hx_fouling"
OUT.mkdir(parents=True, exist_ok=True)

# Explicit inherited-UA multiplier.
# This is a controlled simulation severity, not a universal
# diagnostic threshold or literal "% fouled" measurement.
FOULING_MULTIPLIER = 0.85


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

# Same deterministic archetype assignment used by the previous
# validation experiments.
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

design_loads = {
    j: c.design_load
    for j, c in template.consumers.items()
}

loads = generate_loads(
    archetypes,
    design_loads,
    times,
    cfg.loads,
    rng,
)


# ============================================================
# 2. HEALTHY INSTALLED UA
#
# Use a deterministic healthy sizing factor for this controlled
# experiment. The SAME installed healthy UA is the basis for the
# fouled counterfactual.
# ============================================================

HEALTHY_FACTOR = 1.05

ua_healthy = {
    j: installed_ua(
        design_loads[j],
        cfg,
        healthy_factor=HEALTHY_FACTOR,
        fouling=1.0,
    )
    for j in design_loads
}


# ============================================================
# 3. HEALTHY REFERENCE
# ============================================================

dcn_healthy = build_dcn(net, cfg)

print("Running HEALTHY scenario...")

healthy = run_eps_ntu_scenario(
    dcn_healthy,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua_healthy,
    pipe_U=heat_gain.U,
)

print("✓ Healthy scenario complete")


# ============================================================
# 4. SELECT A HEALTHY, DYNAMIC TARGET
#
# Require zero healthy unmet timesteps, then choose the terminal
# with the largest desired-flow variation.
# ============================================================

candidates = []

for j in healthy.desired_flow.columns:
    unmet_count = int(healthy.unmet[j].sum())
    desired_std = float(healthy.desired_flow[j].std())

    candidates.append(
        (str(j), unmet_count, desired_std)
    )

target, target_healthy_unmet, target_std = min(
    candidates,
    key=lambda x: (x[1], -x[2]),
)

print("Selected target consumer:", target)
print(
    "Healthy target unmet steps:",
    target_healthy_unmet,
    "/",
    n_steps,
)
print(
    "Healthy desired-flow std:",
    target_std,
)

healthy_feasible_mask = (
    ~healthy.unmet[target].to_numpy()
)

print(
    "Healthy-feasible comparison steps:",
    int(healthy_feasible_mask.sum()),
    "/",
    n_steps,
)

if not np.any(healthy_feasible_mask):
    raise RuntimeError(
        "Selected terminal has no healthy-feasible timesteps; "
        "fouling comparison would not be interpretable."
    )


# ============================================================
# 5. FOULED COUNTERFACTUAL
#
# All consumers retain identical UA except the selected target.
# Fouling composes on its installed healthy UA.
# ============================================================

ua_fouled = dict(ua_healthy)

ua_fouled[target] = (
    ua_healthy[target]
    * FOULING_MULTIPLIER
)

dcn_fouled = build_dcn(net, cfg)

print("\nRunning FOULED scenario...")

fouled = run_eps_ntu_scenario(
    dcn_fouled,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua_fouled,
    pipe_U=heat_gain.U,
)

print("✓ Fouled scenario complete")


# ============================================================
# 6. EXTRACT TARGET RESPONSE
# ============================================================

ua_h = float(ua_healthy[target])
ua_f = float(ua_fouled[target])

desired_h = healthy.desired_flow[target].to_numpy()
desired_f = fouled.desired_flow[target].to_numpy()

flow_h = healthy.consumer_flow[target].to_numpy()
flow_f = fouled.consumer_flow[target].to_numpy()

load_h = healthy.delivered_load[target].to_numpy()
load_f = fouled.delivered_load[target].to_numpy()

ets_return_h = healthy.ets_return[target].to_numpy()
ets_return_f = fouled.ets_return[target].to_numpy()

supply_h = healthy.supply_temp[target].to_numpy()
supply_f = fouled.supply_temp[target].to_numpy()

# Primary-side temperature rise across ETS:
# return - supply for chilled-water cooling.
dt_h = ets_return_h - supply_h
dt_f = ets_return_f - supply_f

ua_ratio = ua_f / ua_h

mean_desired_h = float(np.mean(desired_h))
mean_desired_f = float(np.mean(desired_f))

mean_flow_h = float(np.mean(flow_h))
mean_flow_f = float(np.mean(flow_f))

mean_dt_h = float(np.mean(dt_h))
mean_dt_f = float(np.mean(dt_f))

max_desired_change = float(
    np.max(np.abs(desired_f - desired_h))
)

max_realized_change = float(
    np.max(np.abs(flow_f - flow_h))
)

max_load_change = float(
    np.max(np.abs(load_f - load_h))
)

healthy_unmet = int(
    healthy.unmet[target].sum()
)

fouled_unmet = int(
    fouled.unmet[target].sum()
)


# ============================================================
# 7. REPORT
# ============================================================

print("\n" + "=" * 72)
print("EXPERIMENT 05 — HX FOULING VALIDATION")
print("=" * 72)

print("Target consumer:", target)
print("Scenario seed:", scenario_seed)
print("Healthy sizing factor:", HEALTHY_FACTOR)
print("Fouling multiplier:", FOULING_MULTIPLIER)

print("\nINSTALLED UA")
print(
    "Healthy UA [W/K]:",
    f"{ua_h:.12g}",
)
print(
    "Fouled UA [W/K]:",
    f"{ua_f:.12g}",
)
print(
    "Observed UA ratio:",
    f"{ua_ratio:.12g}",
)

print("\nCONTROLLER RESPONSE")
print(
    "Healthy mean desired flow [m3/s]:",
    f"{mean_desired_h:.12g}",
)
print(
    "Fouled mean desired flow [m3/s]:",
    f"{mean_desired_f:.12g}",
)
print(
    "Max |desired-flow change| [m3/s]:",
    f"{max_desired_change:.12g}",
)

print("\nREALIZED FLOW")
print(
    "Healthy mean flow [m3/s]:",
    f"{mean_flow_h:.12g}",
)
print(
    "Fouled mean flow [m3/s]:",
    f"{mean_flow_f:.12g}",
)
print(
    "Max |realized-flow change| [m3/s]:",
    f"{max_realized_change:.12g}",
)

print("\nPRIMARY-SIDE DELTA-T")
print(
    "Healthy mean ΔT [K]:",
    f"{mean_dt_h:.12g}",
)
print(
    "Fouled mean ΔT [K]:",
    f"{mean_dt_f:.12g}",
)

print("\nDELIVERED LOAD")
print(
    "Max |healthy-fouled load difference| [W]:",
    f"{max_load_change:.12g}",
)
print(
    "Healthy unmet steps:",
    healthy_unmet,
)
print(
    "Fouled unmet steps:",
    fouled_unmet,
)

print("\nCONVERGENCE")
print(
    "Healthy:",
    int(healthy.converged.sum()),
    "/",
    len(healthy.converged),
)
print(
    "Fouled:",
    int(fouled.converged.sum()),
    "/",
    len(fouled.converged),
)


# ============================================================
# 8. MECHANISM CHECKS
#
# We deliberately do NOT require fouling to cause unmet load.
# A controller may compensate for reduced UA by requesting more
# primary flow while sufficient hydraulic/thermal authority remains.
# ============================================================

checks = {
    "healthy-feasible comparison steps exist":
        bool(np.any(healthy_feasible_mask)),

    "UA reduced by requested multiplier":
        np.isclose(
            ua_ratio,
            FOULING_MULTIPLIER,
            rtol=1e-12,
            atol=1e-12,
        ),

    "controller responds to reduced UA":
        max_desired_change > 1e-8,

    "realized hydraulic state responds":
        max_realized_change > 1e-8,

    "healthy simulation converged":
        bool(np.all(healthy.converged)),

    "fouled simulation converged":
        bool(np.all(fouled.converged)),
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
# 9. SAVE PLOT-READY RESULTS
# ============================================================

df = pd.DataFrame(
    {
        "time_s": healthy.time,
        "healthy_desired_flow_m3s": desired_h,
        "fouled_desired_flow_m3s": desired_f,
        "healthy_realized_flow_m3s": flow_h,
        "fouled_realized_flow_m3s": flow_f,
        "healthy_delivered_load_W": load_h,
        "fouled_delivered_load_W": load_f,
        "healthy_supply_temp_C": supply_h,
        "fouled_supply_temp_C": supply_f,
        "healthy_ets_return_C": ets_return_h,
        "fouled_ets_return_C": ets_return_f,
        "healthy_deltaT_K": dt_h,
        "fouled_deltaT_K": dt_f,
        "healthy_unmet": healthy.unmet[target].to_numpy(),
        "fouled_unmet": fouled.unmet[target].to_numpy(),
    }
)

df["healthy_UA_WK"] = ua_h
df["fouled_UA_WK"] = ua_f
df["fouling_multiplier"] = FOULING_MULTIPLIER

out_path = (
    OUT /
    "hx_fouling_validation.csv"
)

df.to_csv(out_path, index=False)

print("\nSaved:", out_path)
