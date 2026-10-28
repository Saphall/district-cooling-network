"""Static network context (once per run) and per-seed exogenous context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import district_cooling_network.bootstrap  # noqa: F401  (must precede dcngen imports)
from dcngen.config import load_config
from dcngen.loads.archetypes import ARCHETYPE_NAMES
from dcngen.loads.load_generator import generate_loads
from dcngen.orchestrate.scenario import run_eps_ntu_scenario
from dcngen.thermal.ets import design_ua
from dcngen.thermal.heat_gain import resolve_pipe_heat_gain
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network

from district_cooling_network.paths import Workspace, get_workspace


@dataclass
class StaticContext:
    cfg: Any
    spec: dict[str, Any]
    net: Any
    consumers: list[str]
    archetypes: dict[str, str]
    design_loads: dict[str, float]
    template: Any
    times: np.ndarray


@dataclass
class SeedContext:
    scenario_seed: int
    loads: dict[str, np.ndarray]
    pipe_U: dict[str, float]
    ua_healthy: dict[str, float]
    target: str


def build_static_context(
    spec: dict[str, Any], static_seed: int, ws: Workspace | None = None
) -> StaticContext:
    ws = ws or get_workspace()
    cfg = load_config(ws.dcngen_config())
    n_steps = int(spec["n_steps"])
    times = np.arange(n_steps) * cfg.solver.dt

    net = load_ditec_network(Path(cfg.paths.ditec_data) / cfg.poc.network, cfg.poc.static_scenario_id)
    consumers = sorted(j for j in net.junctions if net.draw.base_demand.get(j, 0.0) > 0.0)

    archetype_rng = np.random.default_rng(static_seed)
    archetypes = {
        j: str(a)
        for j, a in zip(
            consumers, archetype_rng.choice(ARCHETYPE_NAMES, size=len(consumers)), strict=True
        )
    }

    template = build_dcn(net, cfg)
    design_loads = {j: c.design_load for j, c in template.consumers.items()}
    return StaticContext(cfg, spec, net, consumers, archetypes, design_loads, template, times)


def build_seed_context(
    static: StaticContext, scenario_seed: int, cached_target: str | None = None
) -> SeedContext:
    """Exogenous realisation shared by every healthy/fault twin of this seed.

    ``cached_target`` skips the reference run that selects the target ETS.
    """
    cfg, spec = static.cfg, static.spec
    n_steps = int(spec["n_steps"])
    rng = np.random.default_rng(scenario_seed)

    heat_gain = resolve_pipe_heat_gain(static.template, cfg, rng)
    loads = generate_loads(static.archetypes, static.design_loads, static.times, cfg.loads, rng)
    ua_healthy = {
        j: design_ua(static.design_loads[j], cfg) * float(spec["healthy_ua_factor"])
        for j in static.consumers
    }
    pipe_U = dict(heat_gain.U)

    target = cached_target
    if target is None:
        reference = run_eps_ntu_scenario(
            build_dcn(static.net, cfg),
            cfg,
            loads,
            n_steps=n_steps,
            ua=ua_healthy,
            pipe_U=pipe_U,
        )
        # Pick the ETS whose controller request varies most.
        desired_std = {
            j: float(np.std(reference.desired_flow[j].to_numpy(dtype=float)))
            for j in static.consumers
        }
        target = max(desired_std, key=desired_std.get)

    return SeedContext(scenario_seed, loads, pipe_U, ua_healthy, target)
