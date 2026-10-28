
"""Experiment 02 — matched healthy vs excessive plant control margin."""

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
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network


WS = get_workspace()
CFG_PATH = WS.dcngen_config()
OUT = WS.validation_dir("02_excess_control_margin")
OUT.mkdir(parents=True, exist_ok=True)

PUMP_CURVE = "plant_pump_curve"

# Simulation sensitivity parameter — not a universal fault threshold.
HEAD_MULTIPLIER = 1.20

cfg = load_config(CFG_PATH)

n_steps = 48
times = np.arange(n_steps) * cfg.solver.dt


# ------------------------------------------------------------
# 1. Generate ONE common physical scenario
# ------------------------------------------------------------

master_rng = np.random.default_rng(cfg.seed)

static_seed = int(master_rng.integers(2**63))
scenario_seed = int(master_rng.integers(2**63))

net = load_ditec_network(
    Path(cfg.paths.ditec_data) / cfg.poc.network,
    cfg.poc.static_scenario_id,
)

consumers = sorted(
    j for j in net.junctions
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

heat_gain = resolve_pipe_heat_gain(template, cfg, rng)

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


# ------------------------------------------------------------
# 2. HEALTHY
# ------------------------------------------------------------

dcn_h = build_dcn(net, cfg)

healthy_curve = dcn_h.wn.get_curve(PUMP_CURVE)

if len(healthy_curve.points) != 1:
    raise RuntimeError(
        "Expected single-point plant HEAD curve"
    )

q_design_h, h0 = healthy_curve.points[0]

q_design_h = float(q_design_h)
h0 = float(h0)

print("Healthy plant design head H0:", h0, "m")
print("Plant design flow:", q_design_h, "m3/s")

print("\nRunning HEALTHY scenario...")

healthy = run_eps_ntu_scenario(
    dcn_h,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    pipe_U=heat_gain.U,
)

print("✓ Healthy complete")


# ------------------------------------------------------------
# 3. EXCESSIVE CONTROL MARGIN
# ------------------------------------------------------------

dcn_f = build_dcn(net, cfg)

fault_curve = dcn_f.wn.get_curve(PUMP_CURVE)

q_design_f, h0_fault_copy = fault_curve.points[0]

q_design_f = float(q_design_f)
h0_fault_copy = float(h0_fault_copy)

# Verify both fresh networks start identically.
if not np.isclose(h0, h0_fault_copy):
    raise RuntimeError(
        "Healthy and fault DCNs do not have identical H0"
    )

h_fault = h0 * HEAD_MULTIPLIER

fault_curve.points = [
    (q_design_f, float(h_fault))
]

print("\nExcess-head multiplier:", HEAD_MULTIPLIER)
print("Fault plant head:", h_fault, "m")
print("Added head:", h_fault - h0, "m")

print("\nRunning EXCESS-CONTROL-MARGIN scenario...")

fault = run_eps_ntu_scenario(
    dcn_f,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    pipe_U=heat_gain.U,
)

print("✓ Fault scenario complete")


# ------------------------------------------------------------
# 4. Hydraulic pumping calculation
#
# P_hyd = rho g Q H
#
# This is kept separate from result.plant_power.
# ------------------------------------------------------------

# Water density for derived hydraulic-power calculation.
# Not a fault threshold or simulator intervention parameter.
rho = cfg.water.rho  # kg/m3, from simulator configuration
g = 9.80665

q_h = np.asarray(healthy.plant_flow, dtype=float)
q_f = np.asarray(fault.plant_flow, dtype=float)

p_h = rho * g * np.abs(q_h) * h0
p_f = rho * g * np.abs(q_f) * h_fault


# ------------------------------------------------------------
# 5. Counterfactual comparison
# ------------------------------------------------------------

healthy_flow = healthy.consumer_flow.to_numpy()
fault_flow = fault.consumer_flow.to_numpy()

max_consumer_flow_difference = float(
    np.max(np.abs(fault_flow - healthy_flow))
)

max_plant_flow_difference = float(
    np.max(np.abs(q_f - q_h))
)

mean_hydraulic_power_h = float(np.mean(p_h))
mean_hydraulic_power_f = float(np.mean(p_f))

power_ratio = (
    mean_hydraulic_power_f /
    mean_hydraulic_power_h
)

head_ratio = h_fault / h0


# Pressure response
ps_h = healthy.pressure_s.to_numpy()
ps_f = fault.pressure_s.to_numpy()

pr_h = healthy.pressure_r.to_numpy()
pr_f = fault.pressure_r.to_numpy()

mean_supply_pressure_shift = float(
    np.mean(ps_f - ps_h)
)

mean_return_pressure_shift = float(
    np.mean(pr_f - pr_h)
)


# ------------------------------------------------------------
# 6. Save plot-ready summary
# ------------------------------------------------------------

df = pd.DataFrame({
    "time_s": times,
    "healthy_plant_flow_m3s": q_h,
    "fault_plant_flow_m3s": q_f,
    "healthy_head_m": np.full(n_steps, h0),
    "fault_head_m": np.full(n_steps, h_fault),
    "healthy_hydraulic_power_W": p_h,
    "fault_hydraulic_power_W": p_f,
})

df["head_excess_m"] = (
    df["fault_head_m"] -
    df["healthy_head_m"]
)

df["hydraulic_power_excess_W"] = (
    df["fault_hydraulic_power_W"] -
    df["healthy_hydraulic_power_W"]
)

csv_path = OUT / "excess_control_margin_validation.csv"
df.to_csv(csv_path, index=False)


# ------------------------------------------------------------
# 7. Mechanism checks
# ------------------------------------------------------------

tol = 1e-10

checks = {
    "fault head exceeds healthy head":
        h_fault > h0,

    "requested head multiplier applied":
        np.isclose(head_ratio, HEAD_MULTIPLIER),

    "consumer flow field remains matched":
        max_consumer_flow_difference < tol,

    "plant flow remains matched":
        max_plant_flow_difference < tol,

    "hydraulic pumping requirement increases":
        mean_hydraulic_power_f > mean_hydraulic_power_h,

    "healthy simulation converged":
        bool(np.all(healthy.converged)),

    "fault simulation converged":
        bool(np.all(fault.converged)),
}


print("\n" + "=" * 72)
print("EXPERIMENT 02 — EXCESSIVE CONTROL MARGIN VALIDATION")
print("=" * 72)

print("\nHEAD")
print("Healthy H0 [m]:", f"{h0:.6f}")
print("Fault H [m]:", f"{h_fault:.6f}")
print("Head multiplier:", f"{head_ratio:.6f}")

print("\nFLOW")
print(
    "max consumer-flow difference [m3/s]:",
    f"{max_consumer_flow_difference:.12g}",
)
print(
    "max plant-flow difference [m3/s]:",
    f"{max_plant_flow_difference:.12g}",
)

print("\nPRESSURE")
print(
    "mean supply-pressure shift [m]:",
    f"{mean_supply_pressure_shift:.6f}",
)
print(
    "mean return-pressure shift [m]:",
    f"{mean_return_pressure_shift:.6f}",
)

print("\nHYDRAULIC PUMPING")
print(
    "Healthy mean P_hyd [W]:",
    f"{mean_hydraulic_power_h:.3f}",
)
print(
    "Fault mean P_hyd [W]:",
    f"{mean_hydraulic_power_f:.3f}",
)
print(
    "Power ratio:",
    f"{power_ratio:.6f}",
)

print("\nCONVERGENCE")
print(
    "Healthy:",
    int(np.sum(healthy.converged)),
    "/",
    n_steps,
)
print(
    "Fault:",
    int(np.sum(fault.converged)),
    "/",
    n_steps,
)

print("\nMECHANISM CHECKS")

for name, passed in checks.items():
    print(
        f"{'PASS' if passed else 'FAIL':4s}  {name}"
    )

print(
    "\nOVERALL:",
    "PASS" if all(checks.values())
    else "CHECK RESULTS"
)

print("\nSaved:", csv_path)
