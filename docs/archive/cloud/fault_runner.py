
import numpy as np

from dcngen.topology.dcnifier import build_dcn
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.faults.injector import (
    StuckValveInjection,
    inject_poor_balancing,
)
from dcngen.faults.taxonomy import FaultLabel, FaultKind


def run_fault(
    net, cfg, loads, ua_healthy, pipe_U,
    times, target, fsed, severity,
):
    """Run one fault scenario using existing DCN physics."""

    n_steps = len(times)
    dcn = build_dcn(net, cfg)
    kwargs = {}

    if fsed == "stuck_actuator":
        onset_idx = int(round(severity * (n_steps - 1)))
        onset_idx = max(1, min(onset_idx, n_steps - 1))
        onset_s = float(times[onset_idx])

        label = FaultLabel(
            kind=FaultKind.ETS_VALVE,
            location=target,
            element="node",
            side=None,
            severity=1.0,
            onset=onset_s,
            offset=None,
            mask=times >= onset_s,
        )

        kwargs["stuck_valve"] = StuckValveInjection(
            label=label,
            consumer=target,
        )

    elif fsed == "poor_balancing":
        kwargs["poor_balancing"] = inject_poor_balancing(
            dcn, cfg, times,
            consumer=target,
            flow_multiplier=float(severity),
        )

    elif fsed == "excess_control_margin":
        q_design, head = dcn.pump_design
        dcn.wn.get_curve("plant_pump_curve").points = [
            (float(q_design), float(head * severity))
        ]

    elif fsed == "hx_fouling":
        ua_healthy = dict(ua_healthy)
        ua_healthy[target] *= float(severity)

    else:
        raise ValueError(
            f"Unsupported fault: {fsed}. "
            "Clogged filter requires a separate paired-network runner."
        )

    result = run_eps_ntu_scenario(
        dcn,
        cfg,
        loads,
        n_steps=n_steps,
        ua=ua_healthy,
        pipe_U=pipe_U,
        **kwargs,
    )

    return result


def insert_filter(dcn, consumer_name, k_loss):
    """Insert localized hydraulic resistance before the ETS FCV."""
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
        length=0.01,
        diameter=diameter,
        roughness=100.0,
        minor_loss=float(k_loss),
    )

    wn.add_valve(
        valve_name,
        filter_node,
        return_node,
        diameter=diameter,
        valve_type="FCV",
        minor_loss=0.0,
        initial_setting=float(valve.initial_setting),
    )

    dcn.edge_kind[filter_link] = "supply_pipe"
    dcn.edge_kind[valve_name] = "ets"

    return filter_link


def run_filter_pair(
    net, cfg, loads, ua_healthy, pipe_U,
    target, k_clogged=20.0, n_steps=48,
):
    """Run matched clean-filter and clogged-filter scenarios."""
    from dcngen.topology.dcnifier import _size_plant

    pump_curve = "plant_pump_curve"
    filter_u = 1e-6

    # Healthy network: clean filter (K=1).
    clean_dcn = build_dcn(net, cfg)
    clean_link = insert_filter(clean_dcn, target, 1.0)

    q_design, _ = clean_dcn.pump_design
    clean_dcn.wn.get_curve(pump_curve).points = [
        (float(q_design), 1.0e4)
    ]

    _size_plant(clean_dcn, cfg)

    healthy_q, healthy_head = clean_dcn.pump_design

    reservoir_head = float(
        clean_dcn.wn.get_node(
            clean_dcn.plant.reservoir
        ).base_head
    )

    clean_pipe_u = dict(pipe_U)
    clean_pipe_u[clean_link] = filter_u

    healthy = run_eps_ntu_scenario(
        clean_dcn,
        cfg,
        loads,
        n_steps=n_steps,
        ua=ua_healthy,
        pipe_U=clean_pipe_u,
    )

    # Faulted network: clogged filter, identical pump design.
    clogged_dcn = build_dcn(net, cfg)
    clogged_link = insert_filter(
        clogged_dcn, target, float(k_clogged)
    )

    clogged_dcn.pump_design = (
        float(healthy_q),
        float(healthy_head),
    )

    clogged_dcn.wn.get_curve(pump_curve).points = [
        (float(healthy_q), float(healthy_head))
    ]

    clogged_dcn.wn.get_node(
        clogged_dcn.plant.reservoir
    ).base_head = reservoir_head

    clogged_pipe_u = dict(pipe_U)
    clogged_pipe_u[clogged_link] = filter_u

    fault = run_eps_ntu_scenario(
        clogged_dcn,
        cfg,
        loads,
        n_steps=n_steps,
        ua=ua_healthy,
        pipe_U=clogged_pipe_u,
    )

    return healthy, fault
