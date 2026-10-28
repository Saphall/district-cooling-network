
import numpy as np

from dcngen.topology.dcnifier import build_dcn
from dcngen.loads.load_generator import generate_loads
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.thermal.ets import design_ua


def make_scenario_inputs(
    net, cfg, archetypes, design_loads,
    times, seed, healthy_factor,
):
    """Create reproducible dynamic inputs for one scenario seed."""

    rng = np.random.default_rng(int(seed))

    template = build_dcn(net, cfg)

    heat_gain = resolve_pipe_heat_gain(
        template, cfg, rng
    )

    loads = generate_loads(
        archetypes,
        design_loads,
        times,
        cfg.loads,
        rng,
    )

    ua_healthy = {
        consumer: design_ua(design_loads[consumer], cfg)
        * float(healthy_factor)
        for consumer in archetypes
    }

    return {
        "loads": loads,
        "ua_healthy": ua_healthy,
        "pipe_U": dict(heat_gain.U),
    }
