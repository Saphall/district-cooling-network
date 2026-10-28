"""Clogged-filter mechanism: a short passive minor-loss link ahead of the ETS FCV."""

from __future__ import annotations

import district_cooling_network.bootstrap  # noqa: F401  (must precede dcngen imports)


def insert_filter(dcn, consumer_name: str, k_loss: float, *, length_m: float) -> str:
    """Insert a localized resistance immediately upstream of the target ETS FCV.

    ``k_loss`` is a simulation sensitivity parameter. Returns the filter link name.
    """
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
    wn.add_junction(filter_node, base_demand=0.0, elevation=elevation)
    wn.add_pipe(
        filter_link,
        supply_node,
        filter_node,
        length=length_m,
        diameter=diameter,
        roughness=100.0,
        minor_loss=float(k_loss),
    )

    # Re-create the inherited ideal FCV downstream of the filter.
    wn.add_valve(
        valve_name,
        filter_node,
        return_node,
        diameter=diameter,
        valve_type="FCV",
        minor_loss=0.0,
        initial_setting=float(valve.initial_setting),
    )

    # Thermal graph bookkeeping.
    dcn.edge_kind[filter_link] = "supply_pipe"
    dcn.edge_kind[valve_name] = "ets"

    return filter_link
