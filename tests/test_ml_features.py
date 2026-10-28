import numpy as np
import pandas as pd
import pytest

from district_cooling_network.ml.features import add_labels, feature_matrix


def frame():
    t = np.arange(4) * 600.0
    return pd.concat(
        [
            pd.DataFrame({"state": "healthy", "fsed": "stuck_actuator", "time_s": t, "fault_onset_s": 1200.0}),
            pd.DataFrame({"state": "fault", "fsed": "stuck_actuator", "time_s": t, "fault_onset_s": 1200.0}),
            pd.DataFrame({"state": "fault", "fsed": "hx_fouling", "time_s": t, "fault_onset_s": np.nan}),
        ],
        ignore_index=True,
    )


def test_stuck_actuator_is_faulty_only_from_onset():
    out = add_labels(frame())
    stuck_fault = out[(out.state == "fault") & (out.fsed == "stuck_actuator")]
    assert stuck_fault["fault_active"].tolist() == [0, 0, 1, 1]
    assert (out[out.state == "healthy"]["fault_active"] == 0).all()
    fouling = out[out.fsed == "hx_fouling"]
    assert (fouling["fault_active"] == 1).all() and (fouling["fault_class"] == "hx_fouling").all()
    assert (out.loc[out.fault_active == 0, "fault_class"] == "normal").all()


def test_feature_matrix_rejects_leaky_columns():
    with pytest.raises(ValueError, match="leaky"):
        feature_matrix(pd.DataFrame({"fsed": ["x"]}), ("fsed",))
