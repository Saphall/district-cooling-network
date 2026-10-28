
"""Experiment 03 — matched healthy vs poor-balancing scenario."""

from pathlib import Path

import numpy as np
import pandas as pd

from dcngen.config import load_config
from dcngen.faults.injector import inject_poor_balancing
from dcngen.loads.archetypes import ARCHETYPE_NAMES
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import design_ua
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network


ROOT = Path("/content/drive/MyDrive/DCN_Thesis")
CFG_PATH = ROOT / "configs/colab.yaml"
OUT = ROOT / "results/03_poor_balancing"
OUT.mkdir(parents=True, exist_ok=True)

# Simulation sensitivity parameter.
# 0.80 means the affected ETS actuator targets 80% of the
# healthy controller-required flow.
FLOW_MULTIPLIER = 0.80

cfg = load_config(CFG_PATH)

n_steps = 48
times = np.arange(n_steps) * cfg.solver.dt

# ------------------------------------------------------------
# 1. Common exogenous scenario
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

# ------------------------------------------------------------
# 2. Healthy counterfactual
# ------------------------------------------------------------

dcn_h = build_dcn(net, cfg)

print("Running HEALTHY scenario...")

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
# 3. Select a terminal whose required flow actually changes
# ------------------------------------------------------------

desired_std = healthy.desired_flow.std(axis=0)

target = str(desired_std.idxmax())

print("Selected target consumer:", target)
print(
    "Healthy desired-flow std:",
    float(desired_std[target]),
)

# ------------------------------------------------------------
# 4. Poor-balancing intervention
# ------------------------------------------------------------

dcn_f = build_dcn(net, cfg)

balancing = inject_poor_balancing(
    dcn_f,
    cfg,
    times,
    consumer=target,
    flow_multiplier=FLOW_MULTIPLIER,
)

print(
    "Applied flow multiplier:",
    FLOW_MULTIPLIER,
)

print("\nRunning POOR-BALANCING scenario...")

fault = run_eps_ntu_scenario(
    dcn_f,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    poor_balancing=balancing,
    pipe_U=heat_gain.U,
)

print("✓ Poor-balancing scenario complete")

# ------------------------------------------------------------
# 5. Extract target terminal
# ------------------------------------------------------------

q_required = fault.desired_flow[target].to_numpy(
    dtype=float
)

q_applied = fault.consumer_flow[target].to_numpy(
    dtype=float
)

q_healthy = healthy.consumer_flow[target].to_numpy(
    dtype=float
)

q_expected_target = (
    FLOW_MULTIPLIER * q_required
)

# Avoid division by tiny values when calculating normalized residual.
valid = np.abs(q_required) > 1e-12

normalized_residual = np.full(
    n_steps,
    np.nan,
    dtype=float,
)

normalized_residual[valid] = (
    q_applied[valid] - q_required[valid]
) / q_required[valid]

tracking_error = (
    q_applied - q_expected_target
)

# ------------------------------------------------------------
# 6. Diagnostics
# ------------------------------------------------------------

applied_range = float(
    np.max(q_applied) - np.min(q_applied)
)

required_range = float(
    np.max(q_required) - np.min(q_required)
)

max_target_tracking_error = float(
    np.max(np.abs(tracking_error))
)

mean_balancing_residual = float(
    np.nanmean(normalized_residual)
)

max_healthy_fault_target_difference = float(
    np.max(np.abs(q_applied - q_healthy))
)

# Compare unaffected terminals too.
other_consumers = [
    j for j in healthy.consumer_flow.columns
    if j != target
]

if other_consumers:
    unaffected_difference = float(
        np.max(
            np.abs(
                fault.consumer_flow[other_consumers].to_numpy()
                - healthy.consumer_flow[other_consumers].to_numpy()
            )
        )
    )
else:
    unaffected_difference = 0.0

# ------------------------------------------------------------
# 7. Save plot-ready output
# ------------------------------------------------------------

df = pd.DataFrame({
    "time_s": times,
    "required_flow_m3s": q_required,
    "applied_flow_m3s": q_applied,
    "healthy_applied_flow_m3s": q_healthy,
    "biased_target_flow_m3s": q_expected_target,
    "normalized_balancing_residual": normalized_residual,
    "target_tracking_error_m3s": tracking_error,
})

csv_path = OUT / "poor_balancing_validation.csv"

df.to_csv(
    csv_path,
    index=False,
)

# ------------------------------------------------------------
# 8. Mechanism checks
# ------------------------------------------------------------

# Validation tolerances account for the simulator's iterative
# relaxation/convergence process. They are numerical validation
# tolerances, not physical fault thresholds.
max_target_scale = float(
    np.max(np.abs(q_expected_target))
)

relative_tracking_error = (
    max_target_tracking_error / max_target_scale
    if max_target_scale > 0.0
    else 0.0
)

expected_residual = FLOW_MULTIPLIER - 1.0
residual_error = abs(
    mean_balancing_residual - expected_residual
)

checks = {
    "required flow changes with operating condition":
        required_range > 0.0,

    "applied flow remains dynamic":
        applied_range > 0.0,

    "applied flow tracks biased target":
        relative_tracking_error < 1e-3,

    "normalized residual matches imposed bias":
        residual_error < 1e-3,

    "fault changes affected terminal flow":
        max_healthy_fault_target_difference > 0.0,

    "healthy simulation converged":
        bool(np.all(healthy.converged)),

    "fault simulation converged":
        bool(np.all(fault.converged)),
}

print("\n" + "=" * 72)
print("EXPERIMENT 03 — POOR BALANCING VALIDATION")
print("=" * 72)

print("Target consumer:", target)
print("Scenario seed:", scenario_seed)
print(
    "Flow multiplier:",
    FLOW_MULTIPLIER,
)

print("\nREQUIRED FLOW")
print(
    "required-flow range [m3/s]:",
    f"{required_range:.12g}",
)

print("\nAPPLIED FLOW")
print(
    "applied-flow range [m3/s]:",
    f"{applied_range:.12g}",
)

print(
    "max |applied - biased target| [m3/s]:",
    f"{max_target_tracking_error:.12g}",
)

print(
    "relative target-tracking error:",
    f"{relative_tracking_error:.12g}",
)

print("\nBALANCING RESIDUAL")
print(
    "Expected normalized residual:",
    f"{expected_residual:.6f}",
)

print(
    "Mean observed normalized residual:",
    f"{mean_balancing_residual:.6f}",
)

print(
    "Absolute residual error:",
    f"{residual_error:.12g}",
)

print("\nCOUNTERFACTUAL")
print(
    "max |fault target flow - healthy target flow| [m3/s]:",
    f"{max_healthy_fault_target_difference:.12g}",
)

print(
    "max unaffected-consumer flow difference [m3/s]:",
    f"{unaffected_difference:.12g}",
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
    "PASS"
    if all(checks.values())
    else "CHECK RESULTS"
)

print("\nSaved:", csv_path)
