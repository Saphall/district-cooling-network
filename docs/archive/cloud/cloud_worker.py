
from scenario_inputs import make_scenario_inputs
from scenario_generator import generate_pair


def generate_seeded_pair(
    net, cfg, archetypes, design_loads,
    times, target, seed, fsed, severity,
    healthy_factor, output_root,
):
    """Generate one reproducible healthy/fault pair."""

    inputs = make_scenario_inputs(
        net=net,
        cfg=cfg,
        archetypes=archetypes,
        design_loads=design_loads,
        times=times,
        seed=seed,
        healthy_factor=healthy_factor,
    )

    return generate_pair(
        net=net,
        cfg=cfg,
        loads=inputs["loads"],
        ua_healthy=inputs["ua_healthy"],
        pipe_U=inputs["pipe_U"],
        times=times,
        target=target,
        seed=seed,
        fsed=fsed,
        severity=severity,
        output_root=output_root,
    )
