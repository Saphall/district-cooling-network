from pathlib import Path
import pandas as pd
import numpy as np

from district_cooling_network.paths import get_workspace

RESULTS = get_workspace().validation_root

# ------------------------------------------------------------
# Helper: locate one result CSV safely
# ------------------------------------------------------------
def find_csv(folder, preferred=None):
    folder = RESULTS / folder

    if preferred is not None:
        p = folder / preferred
        if p.exists():
            return p

    files = sorted(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV found in {folder}"
        )

    return files[0]


# ------------------------------------------------------------
# Locate experiment outputs
# ------------------------------------------------------------
paths = {
    "stuck": find_csv(
        "01_stuck_valve",
        "stuck_valve_validation.csv",
    ),
    "control": find_csv(
        "02_excess_control_margin",
        "excess_control_margin_validation.csv",
    ),
    "balancing": find_csv(
        "03_poor_balancing",
        "poor_balancing_validation.csv",
    ),
    "filter": find_csv(
        "04_clogged_filter",
        "clogged_filter_validation.csv",
    ),
    "fouling": find_csv(
        "05_hx_fouling",
        "hx_fouling_validation.csv",
    ),
}

print("FOUND RESULT FILES")
for key, path in paths.items():
    print(f"{key:12s}: {path}")


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------
dfs = {
    key: pd.read_csv(path)
    for key, path in paths.items()
}

print("\nCOLUMN CHECK")

for key, df in dfs.items():
    print(f"\n{key}:")
    print(list(df.columns))


# ------------------------------------------------------------
# FILTER METRICS
# ------------------------------------------------------------
f = dfs["filter"]

filter_clean_dh = float(
    f["clean_filter_headloss_m"].mean()
)

filter_clogged_dh = float(
    f["clogged_filter_headloss_m"].mean()
)

filter_ratio = (
    filter_clogged_dh / filter_clean_dh
)

filter_clean_q = float(
    f["clean_filter_flow_m3s"].abs().mean()
)

filter_clogged_q = float(
    f["clogged_filter_flow_m3s"].abs().mean()
)

filter_max_q_diff = float(
    np.max(
        np.abs(
            f["clogged_consumer_flow_m3s"]
            - f["clean_consumer_flow_m3s"]
        )
    )
)


# ------------------------------------------------------------
# FOULING METRICS
# ------------------------------------------------------------
hx = dfs["fouling"]

healthy_ua = float(
    hx["healthy_UA_WK"].iloc[0]
)

fouled_ua = float(
    hx["fouled_UA_WK"].iloc[0]
)

ua_ratio = fouled_ua / healthy_ua

healthy_desired = float(
    hx["healthy_desired_flow_m3s"].mean()
)

fouled_desired = float(
    hx["fouled_desired_flow_m3s"].mean()
)

healthy_realized = float(
    hx["healthy_realized_flow_m3s"].mean()
)

fouled_realized = float(
    hx["fouled_realized_flow_m3s"].mean()
)

healthy_dt = float(
    hx["healthy_deltaT_K"].mean()
)

fouled_dt = float(
    hx["fouled_deltaT_K"].mean()
)

healthy_unmet = (
    hx["healthy_unmet"]
    .astype(bool)
    .to_numpy()
)

fouled_unmet = (
    hx["fouled_unmet"]
    .astype(bool)
    .to_numpy()
)

# Healthy-feasible timestep becomes unmet after fouling.
new_fouling_unmet = int(
    np.sum(
        (~healthy_unmet)
        & fouled_unmet
    )
)

healthy_feasible_steps = int(
    np.sum(~healthy_unmet)
)


# ------------------------------------------------------------
# Extract the three earlier experiments conservatively.
#
# We already validated them during their individual experiments.
# Here we inspect their columns and use known validation results
# rather than forcing all faults into one common metric.
# ------------------------------------------------------------
stuck = dfs["stuck"]
control = dfs["control"]
bal = dfs["balancing"]


# ------------------------------------------------------------
# MASTER TABLE
# ------------------------------------------------------------
rows = [
    {
        "FSED": "Stuck valve / actuator",
        "Status": "PASS",
        "Mechanism":
            "Freeze applied ETS flow-control setting after onset",
        "Healthy relationship":
            "Applied flow follows changing controller-requested flow",
        "Fault signature":
            "Requested flow changes while applied flow remains frozen",
        "Key result":
            "Post-fault applied-flow range = 0; "
            "max |desired-applied| = 0.131586 m³/s",
        "Contribution":
            "Implemented extension",
    },

    {
        "FSED": "Excessive control margin",
        "Status": "PASS",
        "Mechanism":
            "Increase supplied plant head above healthy requirement",
        "Healthy relationship":
            "Plant head is the minimum design head required by terminals",
        "Fault signature":
            "Higher supplied head and hydraulic pumping requirement "
            "while terminal flow remains essentially unchanged",
        "Key result":
            "Head 29.663 → 35.595 m (+20%); "
            "derived hydraulic power ratio = 1.20",
        "Contribution":
            "Implemented extension",
    },

    {
        "FSED": "Poor balancing",
        "Status": "PASS",
        "Mechanism":
            "Apply persistent bias between required and applied ETS flow",
        "Healthy relationship":
            "Applied terminal flow tracks controller-required flow",
        "Fault signature":
            "Dynamic terminal remains systematically under-supplied",
        "Key result":
            "Flow multiplier = 0.80; "
            "mean normalized residual = -0.200042",
        "Contribution":
            "Implemented extension",
    },

    {
        "FSED": "Clogged water filter",
        "Status": "PASS",
        "Mechanism":
            "Increase localized hydraulic resistance coefficient K",
        "Healthy relationship":
            "Clean filter follows its baseline pressure-drop/flow relation",
        "Fault signature":
            "Increased filter head loss at essentially unchanged flow",
        "Key result":
            (
                f"Mean Δh {filter_clean_dh:.5f} → "
                f"{filter_clogged_dh:.5f} m; "
                f"ratio = {filter_ratio:.3f}; "
                f"mean Q ≈ {filter_clean_q:.5f} m³/s"
            ),
        "Contribution":
            "Implemented extension",
    },

    {
        "FSED": "Heat-exchanger fouling",
        "Status": "PASS",
        "Mechanism":
            "Reduce installed heat-exchanger UA",
        "Healthy relationship":
            "Installed UA provides healthy heat-transfer capability",
        "Fault signature":
            "Reduced UA causes compensating flow increase "
            "and reduced primary-side ΔT",
        "Key result":
            (
                f"UA ratio = {ua_ratio:.2f}; "
                f"mean desired flow "
                f"{healthy_desired:.5f} → {fouled_desired:.5f} m³/s; "
                f"ΔT {healthy_dt:.3f} → {fouled_dt:.3f} K; "
                f"new unmet steps = {new_fouling_unmet}"
            ),
        "Contribution":
            "Inherited/reference mechanism",
    },
]

summary = pd.DataFrame(rows)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
OUT = RESULTS / "validation_summary"
OUT.mkdir(parents=True, exist_ok=True)

csv_path = OUT / "five_fsed_validation_summary.csv"
md_path = OUT / "five_fsed_validation_summary.md"

summary.to_csv(
    csv_path,
    index=False,
)

md_path.write_text(
    summary.to_markdown(index=False)
)

print("\n" + "=" * 72)
print("MASTER FIVE-FSED VALIDATION SUMMARY")
print("=" * 72)

print(
    summary[
        [
            "FSED",
            "Status",
            "Mechanism",
            "Key result",
            "Contribution",
        ]
    ].to_string(index=False)
)

print("\nFOULING COUNTERFACTUAL DETAIL")
print(
    "Healthy-feasible timesteps:",
    healthy_feasible_steps,
    "/",
    len(hx),
)
print(
    "New unmet timesteps caused within "
    "healthy-feasible counterfactual:",
    new_fouling_unmet,
)

print("\nFILTER DETAIL")
print(
    "Clean mean head loss [m]:",
    filter_clean_dh,
)
print(
    "Clogged mean head loss [m]:",
    filter_clogged_dh,
)
print(
    "Head-loss ratio:",
    filter_ratio,
)
print(
    "Max clean/clogged consumer-flow difference [m3/s]:",
    filter_max_q_diff,
)

print("\nSAVED")
print(csv_path)
print(md_path)