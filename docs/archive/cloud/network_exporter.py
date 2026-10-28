
import numpy as np
import pandas as pd


def export_network_observations(result, scenario_id, pair_id,
                                state, fsed, target_consumer, seed, onset_s=None):
    """Export every consumer at every simulated timestep."""

    consumers = sorted(result.consumer_flow.columns)
    times = np.asarray(result.time, dtype=float)
    n_steps = len(times)

    channels = {
        "desired_flow_m3s": result.desired_flow,
        "realized_flow_m3s": result.consumer_flow,
        "supply_pressure_m": result.pressure_s,
        "return_pressure_m": result.pressure_r,
        "supply_temp_C": result.supply_temp,
        "return_temp_C": result.return_temp,
        "delivered_load_W": result.delivered_load,
        "unmet": result.unmet,
    }

    frames = []

    for consumer in consumers:
        frame = pd.DataFrame({
            "scenario_id": scenario_id,
            "pair_id": pair_id,
            "state": state,
            "fsed": fsed,
            "target_consumer": str(target_consumer),
            "consumer": str(consumer),
            "seed": int(seed),
            "time_s": times,
            "converged": np.asarray(result.converged, dtype=bool),
        })

        for name, source in channels.items():
            if consumer not in source.columns:
                raise ValueError(
                    f"Missing consumer {consumer} in {name}"
                )
            frame[name] = source[consumer].to_numpy()

        frame["ets_dp_m"] = (
            frame["supply_pressure_m"]
            - frame["return_pressure_m"]
        )

        frame["deltaT_K"] = (
            frame["return_temp_C"]
            - frame["supply_temp_C"]
        )


        if state == "healthy":
            frame["fault_active"] = 0
        elif onset_s is None:
            frame["fault_active"] = 1
        else:
            frame["fault_active"] = (frame["time_s"] - float(times[1] - times[0]) >= float(onset_s) - 1e-9).astype(int)

        frames.append(frame)

    dataset = pd.concat(frames, ignore_index=True)

    expected_rows = n_steps * len(consumers)

    if len(dataset) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, got {len(dataset)}"
        )

    return dataset


def export_pipe_observations(
    result, scenario_id, pair_id, state, fsed,
    target_consumer, seed,
):
    """Export all simulated pipe measurements at every timestep."""
    import numpy as np
    import pandas as pd

    times = np.asarray(result.time, dtype=float)
    pipe_names = sorted(result.pipe_flow.columns)

    frames = []

    for pipe in pipe_names:
        frame = pd.DataFrame({
            "scenario_id": scenario_id,
            "pair_id": pair_id,
            "state": state,
            "fsed": fsed,
            "target_consumer": str(target_consumer),
            "pipe": str(pipe),
            "seed": int(seed),
            "time_s": times,
            "flow_m3s": result.pipe_flow[pipe].to_numpy(),
            "velocity_ms": result.pipe_velocity[pipe].to_numpy(),
            "headloss_m": result.pipe_headloss[pipe].to_numpy(),
        })
        frames.append(frame)

    if not frames:
        raise ValueError("No pipe measurements available")

    output = pd.concat(frames, ignore_index=True)

    expected = len(times) * len(pipe_names)

    if len(output) != expected:
        raise RuntimeError("Pipe export row count mismatch")

    return output
