
"""Experiment 04: clogged-filter hydraulic mechanism validation.

The ETS filter is represented as a localized hydraulic resistance upstream
of the existing ideal FCV.

This experiment compares the same network under:
    clean filter   -> K_clean
    clogged filter -> K_clogged

K values are simulation sensitivity parameters, not diagnostic thresholds
or literal percentages of filter clogging.
"""

from __future__ import annotations

from pathlib import Path

import copy
import math

import numpy as np

from dcngen.config import load_config
from dcngen.topology.dcnifier import build_dcn
from dcngen.topology.ditec_loader import load_ditec_network


CONFIG = (
    "/content/drive/MyDrive/DCN_Thesis/"
    "configs/colab.yaml"
)

# Simulation-defined sensitivity values.
K_CLEAN = 1.0
K_CLOGGED = 20.0

# Numerically short connector:
# distributed pipe friction should be negligible relative to local K loss.
FILTER_LENGTH = 0.01

SEED = 4047939128787533792


def insert_filter(dcn, consumer_name: str, K: float):
    """Insert a passive local-loss element upstream of an ETS FCV."""

    wn = dcn.wn
    consumer = dcn.consumers[consumer_name]

    valve_name = consumer.ets_link
    valve = wn.get_link(valve_name)

    supply_node = valve.start_node_name
    return_node = valve.end_node_name

    diameter = float(valve.diameter)
    setting = float(valve.initial_setting)

    filter_node = f"{valve_name}_filter_node"
    filter_link = f"{valve_name}_filter"

    supply = wn.get_node(supply_node)

    wn.add_junction(
        filter_node,
        base_demand=0.0,
        elevation=float(supply.elevation),
    )

    # Remove original FCV and recreate it downstream of the filter.
    wn.remove_link(valve_name)

    wn.add_pipe(
        filter_link,
        supply_node,
        filter_node,
        length=FILTER_LENGTH,
        diameter=diameter,
        roughness=100.0,
        minor_loss=K,
    )

    wn.add_valve(
        valve_name,
        filter_node,
        return_node,
        diameter=diameter,
        valve_type="FCV",
        initial_setting=setting,
    )

    return {
        "filter_node": filter_node,
        "filter_link": filter_link,
        "supply_node": supply_node,
        "return_node": return_node,
        "diameter": diameter,
    }


def local_loss_head(K: float, q: np.ndarray, diameter: float):
    """Theoretical local-loss head K*v^2/(2g)."""

    area = math.pi * diameter**2 / 4.0
    velocity = q / area

    return K * velocity**2 / (2.0 * 9.80665)


def main():
    cfg = load_config(CONFIG)

    # Load the same inherited DiTEC network used by the
    # already validated experiments.
    net = load_ditec_network(
        Path(cfg.paths.ditec_data) / cfg.poc.network,
        cfg.poc.static_scenario_id,
    )

    clean = build_dcn(net, cfg)
    clogged = build_dcn(net, cfg)

    # Pick a real consumer deterministically.
    target = sorted(clean.consumers)[0]

    clean_meta = insert_filter(
        clean,
        target,
        K_CLEAN,
    )

    clogged_meta = insert_filter(
        clogged,
        target,
        K_CLOGGED,
    )

    print("=" * 72)
    print("EXPERIMENT 04 — CLOGGED FILTER TOPOLOGY CHECK")
    print("=" * 72)

    print("Target consumer:", target)
    print("K clean:", K_CLEAN)
    print("K clogged:", K_CLOGGED)

    print("\nCLEAN FILTER")
    print("filter link:", clean_meta["filter_link"])
    print("filter node:", clean_meta["filter_node"])
    print("diameter [m]:", clean_meta["diameter"])

    clean_filter = clean.wn.get_link(
        clean_meta["filter_link"]
    )

    clogged_filter = clogged.wn.get_link(
        clogged_meta["filter_link"]
    )

    print("\nHYDRAULIC PARAMETERS")
    print(
        "clean minor_loss:",
        clean_filter.minor_loss,
    )
    print(
        "clogged minor_loss:",
        clogged_filter.minor_loss,
    )
    print(
        "clean length [m]:",
        clean_filter.length,
    )

    # Confirm FCV survived and now begins downstream of filter.
    clean_fcv = clean.wn.get_link(
        clean.consumers[target].ets_link
    )

    print("\nTOPOLOGY")
    print(
        clean_filter.start_node_name,
        "->",
        clean_filter.name,
        "->",
        clean_filter.end_node_name,
        "->",
        clean_fcv.name,
        "->",
        clean_fcv.end_node_name,
    )

    checks = {
        "clean filter exists":
            clean_meta["filter_link"]
            in clean.wn.link_name_list,

        "clogged filter exists":
            clogged_meta["filter_link"]
            in clogged.wn.link_name_list,

        "same filter diameter":
            abs(
                clean_meta["diameter"]
                - clogged_meta["diameter"]
            ) < 1e-12,

        "clogged K exceeds clean K":
            clogged_filter.minor_loss
            > clean_filter.minor_loss,

        "FCV retained":
            clean.consumers[target].ets_link
            in clean.wn.link_name_list,

        "filter and FCV are in series":
            clean_filter.end_node_name
            == clean_fcv.start_node_name,
    }

    print("\nTOPOLOGY CHECKS")

    for name, passed in checks.items():
        print(
            "PASS " if passed else "FAIL ",
            name,
        )

    print(
        "\nOVERALL:",
        "PASS" if all(checks.values()) else "FAIL",
    )


if __name__ == "__main__":
    main()
