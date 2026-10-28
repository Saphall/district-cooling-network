from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from district_cooling_network.paths import get_workspace

RESULTS = get_workspace().validation_root
OUT = RESULTS / "validation_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load validated results
# ---------------------------------------------------------
stuck = pd.read_csv(
    RESULTS / "01_stuck_valve/stuck_valve_validation.csv"
)
control = pd.read_csv(
    RESULTS / "02_excess_control_margin/excess_control_margin_validation.csv"
)
bal = pd.read_csv(
    RESULTS / "03_poor_balancing/poor_balancing_validation.csv"
)
filt = pd.read_csv(
    RESULTS / "04_clogged_filter/clogged_filter_validation.csv"
)
foul = pd.read_csv(
    RESULTS / "05_hx_fouling/hx_fouling_validation.csv"
)

def hours(df):
    return df["time_s"] / 3600.0

def save(fig, name):
    png = OUT / f"{name}.png"
    pdf = OUT / f"{name}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print("Saved:", png)
    print("Saved:", pdf)


# =========================================================
# FIGURE 1 — STUCK VALVE
# =========================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

t = hours(stuck)

ax.plot(
    t,
    stuck["fault_desired_flow"],
    label="Controller-requested flow",
    linewidth=2,
)

ax.plot(
    t,
    stuck["fault_applied_flow"],
    label="Applied flow",
    linewidth=2,
)

active = stuck["fault_active"].astype(bool)

if active.any():
    onset = t[active].iloc[0]
    ax.axvline(
        onset,
        linestyle="--",
        linewidth=1.5,
        label="Fault onset",
    )

ax.set_xlabel("Time [h]")
ax.set_ylabel("Primary flow [m³/s]")
ax.set_title("Stuck ETS Flow-Control Actuator")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

save(fig, "01_stuck_valve_validation")
plt.show()


# =========================================================
# FIGURE 2 — EXCESSIVE CONTROL MARGIN
# =========================================================
fig, ax1 = plt.subplots(figsize=(8, 4.5))

t = hours(control)

ax1.plot(
    t,
    control["healthy_head_m"],
    label="Healthy plant head",
    linewidth=2,
)

ax1.plot(
    t,
    control["fault_head_m"],
    label="Excess-margin plant head",
    linewidth=2,
)

ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Plant head [m]")
ax1.grid(alpha=0.25)

ax2 = ax1.twinx()

ax2.plot(
    t,
    control["healthy_hydraulic_power_W"] / 1e6,
    linestyle="--",
    label="Healthy hydraulic power",
)

ax2.plot(
    t,
    control["fault_hydraulic_power_W"] / 1e6,
    linestyle="--",
    label="Excess-margin hydraulic power",
)

ax2.set_ylabel("Hydraulic power [MW]")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
)

ax1.set_title("Excessive Control Margin")

fig.tight_layout()

save(fig, "02_excess_control_margin_validation")
plt.show()


# =========================================================
# FIGURE 3 — POOR BALANCING
# =========================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

t = hours(bal)

ax.plot(
    t,
    bal["required_flow_m3s"],
    label="Required flow",
    linewidth=2,
)

ax.plot(
    t,
    bal["applied_flow_m3s"],
    label="Applied flow",
    linewidth=2,
)

ax.set_xlabel("Time [h]")
ax.set_ylabel("Primary flow [m³/s]")
ax.set_title("Poor Hydronic Balancing")
ax.grid(alpha=0.25)
ax.legend()

fig.tight_layout()

save(fig, "03_poor_balancing_validation")
plt.show()


# =========================================================
# FIGURE 4 — CLOGGED FILTER
# =========================================================
fig, ax1 = plt.subplots(figsize=(8, 4.5))

t = hours(filt)

ax1.plot(
    t,
    filt["clean_filter_headloss_m"],
    label="Clean-filter head loss",
    linewidth=2,
)

ax1.plot(
    t,
    filt["clogged_filter_headloss_m"],
    label="Clogged-filter head loss",
    linewidth=2,
)

ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Filter head loss [m]")
ax1.grid(alpha=0.25)

ax2 = ax1.twinx()

ax2.plot(
    t,
    filt["clean_filter_flow_m3s"],
    linestyle=":",
    label="Clean-filter flow",
)

ax2.plot(
    t,
    filt["clogged_filter_flow_m3s"],
    linestyle="--",
    label="Clogged-filter flow",
)

ax2.set_ylabel("Filter flow [m³/s]")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
)

ax1.set_title("Clogged Water Filter")

fig.tight_layout()

save(fig, "04_clogged_filter_validation")
plt.show()


# =========================================================
# FIGURE 5 — HX FOULING
# =========================================================
fig, ax1 = plt.subplots(figsize=(8, 4.5))

t = hours(foul)

ax1.plot(
    t,
    foul["healthy_desired_flow_m3s"],
    label="Healthy requested flow",
    linewidth=2,
)

ax1.plot(
    t,
    foul["fouled_desired_flow_m3s"],
    label="Fouled requested flow",
    linewidth=2,
)

ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Requested primary flow [m³/s]")
ax1.grid(alpha=0.25)

ax2 = ax1.twinx()

ax2.plot(
    t,
    foul["healthy_deltaT_K"],
    linestyle=":",
    label="Healthy ΔT",
)

ax2.plot(
    t,
    foul["fouled_deltaT_K"],
    linestyle="--",
    label="Fouled ΔT",
)

ax2.set_ylabel("Primary-side ΔT [K]")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best",
)

ua_h = foul["healthy_UA_WK"].iloc[0] / 1e6
ua_f = foul["fouled_UA_WK"].iloc[0] / 1e6

ax1.set_title(
    f"Heat-Exchanger Fouling "
    f"(UA: {ua_h:.2f} → {ua_f:.2f} MW/K)"
)

fig.tight_layout()

save(fig, "05_hx_fouling_validation")
plt.show()


print("\n" + "=" * 65)
print("FIVE-FSED VALIDATION FIGURES COMPLETE")
print("=" * 65)
print("Output directory:")
print(OUT)