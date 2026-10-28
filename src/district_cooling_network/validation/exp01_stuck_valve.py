
"""Experiment 01 — matched healthy vs stuck ETS flow-control actuator."""

from pathlib import Path
import numpy as np
import pandas as pd

import district_cooling_network.bootstrap  # noqa: F401
from district_cooling_network.paths import get_workspace

from dcngen.config import load_config
from dcngen.topology.ditec_loader import load_ditec_network
from dcngen.topology.dcnifier import build_dcn
from dcngen.loads.load_generator import generate_loads
from dcngen.thermal.ets import design_ua
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.faults.injector import StuckValveInjection
from dcngen.faults.taxonomy import FaultLabel, FaultKind


WS = get_workspace()
CFG_PATH = WS.dcngen_config()
OUT = WS.validation_dir("01_stuck_valve")
OUT.mkdir(parents=True, exist_ok=True)

cfg = load_config(CFG_PATH)

# Short validation run first.
# Once physics is verified, longer dataset runs can follow.
n_steps = 48
times = np.arange(n_steps) * cfg.solver.dt

# ------------------------------------------------------------
# 1. Reproduce inherited physical-scenario construction
# ------------------------------------------------------------

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

from dcngen.loads.archetypes import ARCHETYPE_NAMES

archetype_rng = np.random.default_rng(static_seed)

archetypes = {
    j: str(a)
    for j, a in zip(
        consumers,
        archetype_rng.choice(ARCHETYPE_NAMES, size=len(consumers)),
    )
}

# Build one temporary DCN only to generate the common scenario inputs.
dcn_template = build_dcn(net, cfg)

rng = np.random.default_rng(scenario_seed)

heat_gain = resolve_pipe_heat_gain(dcn_template, cfg, rng)

design = {
    j: c.design_load
    for j, c in dcn_template.consumers.items()
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
# 2. HEALTHY counterfactual
# ------------------------------------------------------------

print("Running HEALTHY scenario...")

dcn_h = build_dcn(net, cfg)

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
# 3. Select a useful valve target
#
# We choose the consumer whose healthy desired flow varies most
# after the intended onset. This avoids selecting a nearly
# constant-load terminal where freezing would be uninformative.
# ------------------------------------------------------------

onset_idx = n_steps // 2
onset_time = float(times[onset_idx])

post_std = healthy.desired_flow.iloc[onset_idx:].std()

target = str(post_std.idxmax())

print(f"Selected target consumer: {target}")
print(f"Fault onset: {onset_time:.0f} s")
print(
    "Healthy post-onset desired-flow std:",
    float(post_std[target]),
)

# ------------------------------------------------------------
# 4. Construct stuck-valve label
#
# severity=1.0 is a categorical activation marker.
# It does NOT mean 100% closure or 100% flow reduction.
# ------------------------------------------------------------

mask = np.zeros(n_steps, dtype=bool)
mask[onset_idx:] = True

label = FaultLabel(
    kind=FaultKind.ETS_VALVE,
    location=target,
    element="node",
    side=None,
    severity=1.0,
    onset=onset_time,
    offset=None,
    mask=mask,
)

fault = StuckValveInjection(
    label=label,
    consumer=target,
)

# ------------------------------------------------------------
# 5. FAULTED scenario — fresh DCN, identical exogenous inputs
# ------------------------------------------------------------

print("\nRunning STUCK-VALVE scenario...")

dcn_f = build_dcn(net, cfg)

stuck = run_eps_ntu_scenario(
    dcn_f,
    cfg,
    loads,
    n_steps=n_steps,
    ua=ua,
    stuck_valve=fault,
    pipe_U=heat_gain.U,
)

print("✓ Stuck-valve scenario complete")

# ------------------------------------------------------------
# 6. Build validation table
# ------------------------------------------------------------

df = pd.DataFrame({
    "time_s": times,

    "healthy_desired_flow":
        healthy.desired_flow[target].to_numpy(),

    "healthy_applied_flow":
        healthy.consumer_flow[target].to_numpy(),

    "fault_desired_flow":
        stuck.desired_flow[target].to_numpy(),

    "fault_applied_flow":
        stuck.consumer_flow[target].to_numpy(),

    "healthy_delivered_load":
        healthy.delivered_load[target].to_numpy(),

    "fault_delivered_load":
        stuck.delivered_load[target].to_numpy(),

    "healthy_unmet":
        healthy.unmet[target].to_numpy(),

    "fault_unmet":
        stuck.unmet[target].to_numpy(),
})

df["fault_command_actuation_residual"] = (
    df["fault_desired_flow"]
    - df["fault_applied_flow"]
)

df["counterfactual_flow_difference"] = (
    df["fault_applied_flow"]
    - df["healthy_applied_flow"]
)

df["counterfactual_load_difference"] = (
    df["fault_delivered_load"]
    - df["healthy_delivered_load"]
)

df["fault_active"] = mask

csv_path = OUT / "stuck_valve_validation.csv"
df.to_csv(csv_path, index=False)

# ------------------------------------------------------------
# 7. Quantitative validation checks
# ------------------------------------------------------------

pre = ~mask
post = mask

pre_pair_diff = np.max(
    np.abs(
        df.loc[pre, "fault_applied_flow"]
        - df.loc[pre, "healthy_applied_flow"]
    )
)

post_fault_range = np.ptp(
    df.loc[post, "fault_applied_flow"].to_numpy()
)

post_desired_range = np.ptp(
    df.loc[post, "fault_desired_flow"].to_numpy()
)

post_residual_max = np.max(
    np.abs(
        df.loc[post, "fault_command_actuation_residual"]
    )
)

print("\n" + "=" * 70)
print("EXPERIMENT 01 — STUCK VALVE VALIDATION")
print("=" * 70)

print("Target consumer:", target)
print("Scenario seed:", scenario_seed)
print("Onset [s]:", onset_time)

print("\nPRE-FAULT")
print(
    "max |fault applied - healthy applied|:",
    f"{pre_pair_diff:.12g}",
)

print("\nPOST-FAULT")
print(
    "fault applied-flow range:",
    f"{post_fault_range:.12g}",
)
print(
    "fault desired-flow range:",
    f"{post_desired_range:.12g}",
)
print(
    "max |desired - applied|:",
    f"{post_residual_max:.12g}",
)

print("\nSTATE")
print(
    "Healthy converged steps:",
    int(np.sum(healthy.converged)),
    "/",
    n_steps,
)
print(
    "Fault converged steps:",
    int(np.sum(stuck.converged)),
    "/",
    n_steps,
)
print(
    "Healthy unmet steps:",
    int(np.sum(healthy.unmet[target])),
)
print(
    "Fault unmet steps:",
    int(np.sum(stuck.unmet[target])),
)

print("\nSaved:", csv_path)

# Mechanistic checks — numerical, not universal physical thresholds.
tol = 1e-10

checks = {
    "matched before fault":
        pre_pair_diff < tol,

    "actuator frozen after onset":
        post_fault_range < tol,

    "controller request changes after onset":
        post_desired_range > tol,

    "command-actuation mismatch appears":
        post_residual_max > tol,

    "healthy simulation converged":
        bool(np.all(healthy.converged)),

    "fault simulation converged":
        bool(np.all(stuck.converged)),
}

print("\nMECHANISM CHECKS")
for name, passed in checks.items():
    print(
        f"{'PASS' if passed else 'FAIL':4s}  {name}"
    )

print("\nOVERALL:", "PASS" if all(checks.values()) else "CHECK RESULTS")
