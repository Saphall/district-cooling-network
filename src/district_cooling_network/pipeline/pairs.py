"""One counterfactual pair (healthy twin + faulted twin) per FSED mechanism.

The physics is the legacy Experiment 06 logic, unchanged. Both twins share the
seed's loads and pipe heat gain; each is built from a fresh network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import district_cooling_network.bootstrap  # noqa: F401  (must precede dcngen imports)
from dcngen.faults.injector import StuckValveInjection, inject_poor_balancing
from dcngen.faults.taxonomy import FaultKind, FaultLabel
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.topology.dcnifier import _size_plant, build_dcn

from district_cooling_network.manifest import PairSpec
from district_cooling_network.pipeline.context import SeedContext, StaticContext
from district_cooling_network.pipeline.filter import insert_filter


@dataclass
class PairResult:
    """Raw simulation results of both twins plus what the exporters need."""

    healthy: Any
    fault: Any
    healthy_head: float
    fault_head: float
    onset_s: float | None = None
    healthy_filter: str | None = None
    fault_filter: str | None = None


def _run(static: StaticContext, dcn, sc: SeedContext, *, ua=None, pipe_U=None, **hooks):
    return run_eps_ntu_scenario(
        dcn,
        static.cfg,
        sc.loads,
        n_steps=int(static.spec["n_steps"]),
        ua=sc.ua_healthy if ua is None else ua,
        pipe_U=sc.pipe_U if pipe_U is None else pipe_U,
        **hooks,
    )


def _stuck_actuator(static, sc, pair):
    n_steps = int(static.spec["n_steps"])
    times = static.times
    # Keep onset away from t=0: the stuck state needs a pre-fault actuator setting.
    onset_idx = max(1, min(int(round(pair.severity_value * (n_steps - 1))), n_steps - 1))
    onset_s = float(times[onset_idx])

    healthy_dcn = build_dcn(static.net, static.cfg)
    healthy = _run(static, healthy_dcn, sc)
    head = float(healthy_dcn.pump_design[1])

    label = FaultLabel(
        kind=FaultKind.ETS_VALVE,
        location=sc.target,
        element="node",
        side=None,
        severity=1.0,
        onset=onset_s,
        offset=None,
        mask=times >= onset_s,
    )
    stuck = StuckValveInjection(label=label, consumer=sc.target)
    fault = _run(static, build_dcn(static.net, static.cfg), sc, stuck_valve=stuck)

    return healthy, fault, {"onset_s": onset_s, "healthy_head": head, "fault_head": head}


def _poor_balancing(static, sc, pair):
    healthy_dcn = build_dcn(static.net, static.cfg)
    healthy = _run(static, healthy_dcn, sc)
    head = float(healthy_dcn.pump_design[1])

    fault_dcn = build_dcn(static.net, static.cfg)
    balancing = inject_poor_balancing(
        fault_dcn,
        static.cfg,
        static.times,
        consumer=sc.target,
        flow_multiplier=pair.severity_value,
    )
    fault = _run(static, fault_dcn, sc, poor_balancing=balancing)

    return healthy, fault, {"healthy_head": head, "fault_head": head}


def _excess_control_margin(static, sc, pair):
    curve = static.spec["pump_curve"]

    healthy_dcn = build_dcn(static.net, static.cfg)
    healthy = _run(static, healthy_dcn, sc)
    _, h0 = healthy_dcn.pump_design

    fault_dcn = build_dcn(static.net, static.cfg)
    qf, hf = fault_dcn.pump_design
    if not np.isclose(hf, h0):
        raise RuntimeError("Healthy/fault base plant heads differ.")

    fault_head = float(h0 * pair.severity_value)
    fault_dcn.wn.get_curve(curve).points = [(float(qf), fault_head)]
    fault = _run(static, fault_dcn, sc)

    return healthy, fault, {"healthy_head": float(h0), "fault_head": fault_head}


def _clogged_filter(static, sc, pair):
    flt = static.spec["filter"]
    curve = static.spec["pump_curve"]
    length = float(flt["length_m"])
    u_prime = float(flt["u_prime"])

    # Healthy clean-filter network, with the plant re-sized around the clean filter.
    clean_dcn = build_dcn(static.net, static.cfg)
    clean_filter = insert_filter(clean_dcn, sc.target, float(flt["k_clean"]), length_m=length)

    # Restore the inherited sizing probe before re-sizing.
    q_design, _ = clean_dcn.pump_design
    clean_dcn.wn.get_curve(curve).points = [(float(q_design), 1.0e4)]
    _size_plant(clean_dcn, static.cfg)

    healthy_q_design, healthy_head = clean_dcn.pump_design
    healthy_reservoir_head = float(clean_dcn.wn.get_node(clean_dcn.plant.reservoir).base_head)

    clean_pipe_U = dict(sc.pipe_U)
    clean_pipe_U[clean_filter] = u_prime
    healthy = _run(static, clean_dcn, sc, pipe_U=clean_pipe_U)

    # Faulted network keeps the healthy plant design.
    clogged_dcn = build_dcn(static.net, static.cfg)
    clogged_filter = insert_filter(
        clogged_dcn, sc.target, pair.severity_value, length_m=length
    )
    clogged_dcn.pump_design = (float(healthy_q_design), float(healthy_head))
    clogged_dcn.wn.get_curve(curve).points = [(float(healthy_q_design), float(healthy_head))]
    clogged_dcn.wn.get_node(clogged_dcn.plant.reservoir).base_head = healthy_reservoir_head

    clogged_pipe_U = dict(sc.pipe_U)
    clogged_pipe_U[clogged_filter] = u_prime
    fault = _run(static, clogged_dcn, sc, pipe_U=clogged_pipe_U)

    head = float(healthy_head)
    return healthy, fault, {
        "healthy_head": head,
        "fault_head": head,
        "healthy_filter": clean_filter,
        "fault_filter": clogged_filter,
    }


def _hx_fouling(static, sc, pair):
    healthy_dcn = build_dcn(static.net, static.cfg)
    healthy = _run(static, healthy_dcn, sc)
    head = float(healthy_dcn.pump_design[1])

    ua_fault = dict(sc.ua_healthy)
    ua_fault[sc.target] = sc.ua_healthy[sc.target] * pair.severity_value
    fault = _run(static, build_dcn(static.net, static.cfg), sc, ua=ua_fault)

    return healthy, fault, {"healthy_head": head, "fault_head": head}


_RUNNERS = {
    "stuck_actuator": _stuck_actuator,
    "poor_balancing": _poor_balancing,
    "excess_control_margin": _excess_control_margin,
    "clogged_filter": _clogged_filter,
    "hx_fouling": _hx_fouling,
}


def run_pair(static: StaticContext, sc: SeedContext, pair: PairSpec) -> PairResult:
    """Simulate both twins of one counterfactual pair."""
    healthy, fault, info = _RUNNERS[pair.fsed](static, sc, pair)
    return PairResult(healthy=healthy, fault=fault, **info)
