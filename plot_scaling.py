#!/usr/bin/env python3
"""
Generate scaling-law plots for EgoDex percentages.

Inputs:
- CSV: eval_results/scaling laws egodex - Sheet2.csv
- Total human videos: 314,839
- Average video length: 8.09 seconds

- plots/sr_vs_percentage.png
- plots/sr_vs_num_videos.png
- plots/sr_vs_hours.png
- plots/sr_vs_hours_projection.png
- plots/sr_vs_operator_hours_projection.png
- plots/cost_vs_target_success.png
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser(description="Generate scaling-law plots for EgoDex percentages.")
parser.add_argument(
    "--remove_hours_leq_one",
    action="store_true",
    help="Exclude samples with hours <= 1 when fitting projection curves.",
)
parser.add_argument(
    "--fit_mode",
    choices=["power", "linear", "log"],
    default="power",
    help="Type of curve fit to use for projections (options: power, linear, log).",
)
args = parser.parse_args()

# ---- Configuration ---------------------------------------------------------
CSV_PATH = "eval_results/scaling laws egodex - Sheet2.csv"
BASELINE_PATH = "eval_results/baselines.csv"
ROBOT_DATA_PATH = "eval_results/robot_data_success_rate.csv"
OUTPUT_DIR = "plots"
TOTAL_VIDEOS = 314_839
AVG_VIDEO_SECONDS = 8.09  # average length per video
SCALING_TARGET_HOURS = 1e8  # extrapolation target
EGOVIDEO_TO_OPERATOR = 1.25  # 1 ego hour -> 1.25 operator hours
ROBOTDATA_TO_OPERATOR = 3.0  # 1 robot-data hour -> 3 operator hours (assumption)
OPERATOR_HOUR_COST = 20.0  # USD per operator hour
SCALING_TARGET_OPERATOR = SCALING_TARGET_HOURS * EGOVIDEO_TO_OPERATOR
BASELINE_STYLES = {
    "π0": {"color": "dimgray", "linestyle": "--"},
    "DP": {"color": "saddlebrown", "linestyle": ":"},
}


def fit_curve(x, y, mode):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if mode == "power":
        pos_mask = x > 0
        x = x[pos_mask]
        y = y[pos_mask]
        if x.size < 2:
            return None
        coeffs = np.polyfit(np.log10(x), y, deg=1)
    elif mode == "log":
        pos_mask = x > 0
        x = x[pos_mask]
        y = y[pos_mask]
        if x.size < 2:
            return None
        coeffs = np.polyfit(np.log(x), y, deg=1)
    else:  # linear
        if x.size < 2:
            return None
        coeffs = np.polyfit(x, y, deg=1)
    return coeffs


def evaluate_curve(coeffs, x, mode):
    if coeffs is None:
        return np.array([])
    x = np.asarray(x, dtype=float)
    if mode == "power":
        return np.polyval(coeffs, np.log10(x))
    if mode == "log":
        return np.polyval(coeffs, np.log(x))
    return np.polyval(coeffs, x)


def invert_curve(coeffs, target_y, mode):
    if coeffs is None:
        return np.nan
    a, b = coeffs
    if mode == "power":
        if abs(a) < 1e-10:
            return np.nan
        return 10 ** ((target_y - b) / a)
    if mode == "log":
        if abs(a) < 1e-10:
            return np.nan
        return np.exp((target_y - b) / a)
    if abs(a) < 1e-10:
        return np.nan
    return (target_y - b) / a


def prepare_curve(x_obs, y_obs, target_max, mode, remove_leq_one, min_fit_x=1.0):
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    x_obs = x_obs[mask]
    y_obs = y_obs[mask]

    if mode == "power":
        pos_mask = x_obs > 0
        x_obs = x_obs[pos_mask]
        y_obs = y_obs[pos_mask]
    elif mode == "log":
        pos_mask = x_obs > 0
        x_obs = x_obs[pos_mask]
        y_obs = y_obs[pos_mask]

    scatter_x = x_obs.copy()
    scatter_y = y_obs.copy()

    if x_obs.size == 0:
        return {
            "scatter_x": scatter_x,
            "scatter_y": scatter_y,
            "fit_x": np.array([]),
            "fit_y": np.array([]),
            "coeffs": None,
            "projection_x": np.array([]),
            "projection_y": np.array([]),
            "scatter_used_mask": np.array([], dtype=bool),
        }

    fit_mask = np.ones_like(x_obs, dtype=bool)
    if remove_leq_one:
        threshold = min_fit_x if mode in ("power", "log") else max(min_fit_x, 0.0)
        fit_mask &= x_obs > threshold

    if fit_mask.sum() < 2:
        fit_mask = np.ones_like(x_obs, dtype=bool)

    x_fit = x_obs[fit_mask]
    y_fit = y_obs[fit_mask]

    coeffs = fit_curve(x_fit, y_fit, mode)

    if coeffs is None or target_max <= 0:
        projection_x = np.array([])
        projection_y = np.array([])
    else:
        if mode == "power":
            min_x = max(x_fit.min(), 1e-6)
            if target_max <= min_x:
                projection_x = np.array([])
            else:
                projection_x = np.logspace(np.log10(min_x), np.log10(target_max), num=400)
        elif mode == "log":
            min_x = max(x_fit.min(), 1e-6)
            if target_max <= min_x:
                projection_x = np.array([])
            else:
                projection_x = np.logspace(np.log10(min_x), np.log10(target_max), num=400)
        else:
            min_x = x_fit.min()
            if target_max <= min_x:
                projection_x = np.array([])
            else:
                projection_x = np.linspace(min_x, target_max, num=400)
        projection_y = np.clip(evaluate_curve(coeffs, projection_x, mode), 0.0, 1.0) if projection_x.size else np.array([])

    return {
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
        "fit_x": x_fit,
        "fit_y": y_fit,
        "coeffs": coeffs,
        "projection_x": projection_x,
        "projection_y": projection_y,
        "scatter_used_mask": fit_mask.astype(bool),
    }

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load and preprocess data ----------------------------------------------
# Read CSV
df = pd.read_csv(CSV_PATH)
df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c, inplace=True)

# Baseline data
baselines_df = pd.read_csv(BASELINE_PATH, index_col=0)
baselines_df.rename(columns=lambda c: c.strip(), inplace=True)
baselines_df.index = baselines_df.index.str.strip()
baselines_df = baselines_df.apply(pd.to_numeric, errors='coerce')

# Robot data points for projection plot
robot_df = pd.read_csv(ROBOT_DATA_PATH)
robot_df.columns = [col.strip() for col in robot_df.columns]
robot_df = robot_df.dropna(subset=['Robot Data Hours', 'Success Rate'])
robot_df.loc[robot_df['Robot Data Hours'] <= 0, 'Robot Data Hours'] = 1
robot_hours_raw = robot_df['Robot Data Hours'].astype(float).to_numpy()
robot_success_raw = robot_df['Success Rate'].astype(float).to_numpy()

# Clean percentage column to numeric (0-100)
def parse_pct(value):
    if isinstance(value, str) and value.endswith('%'):
        return float(value.strip('%'))
    return float(value)

df['Egodex data %'] = df['Egodex data %'].apply(parse_pct)

# Compute number of videos and hours
df['Num videos'] = (df['Egodex data %'] / 100.0) * TOTAL_VIDEOS
df['Hours'] = (df['Num videos'] * AVG_VIDEO_SECONDS) / 3600.0

# For log-scale plots, replace zeros with 1 (baseline tick)
df.loc[df['Num videos'] <= 0, 'Num videos'] = 1
df.loc[df['Hours'] <= 0, 'Hours'] = 1

# Melt for plotting (task name column)
tasks = [col for col in df.columns if col not in ['Egodex data %', 'Num videos', 'Hours']]
df_sorted = df.sort_values('Egodex data %').reset_index(drop=True)
df_sorted['Average Success Rate'] = df_sorted[tasks].mean(axis=1)

baseline_means = {
    method: baselines_df.loc[method, tasks].mean()
    for method in baselines_df.index
    if method in BASELINE_STYLES
}

melted = df_sorted.melt(
    id_vars=['Egodex data %', 'Num videos', 'Hours'],
    value_vars=tasks,
    var_name='Task',
    value_name='Success Rate'
)

# Ensure success rates are float
melted['Success Rate'] = melted['Success Rate'].astype(float)

# ---- Plot helpers ----------------------------------------------------------
sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "legend.title_fontsize": 16,
    }
)

def save_plot(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"Saved: {path}")
    plt.close(fig)

def add_baseline_lines(ax, x_values):
    handles = []
    for method, style in BASELINE_STYLES.items():
        if method not in baseline_means:
            continue
        mean_val = baseline_means[method]
        ax.plot(
            x_values,
            [mean_val] * len(x_values),
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=1.3,
            alpha=0.9,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=style['color'],
                linestyle=style['linestyle'],
                label=f"{method} baseline",
                linewidth=1.3,
            )
        )
    return handles

# ---- Plot 1: Success Rate vs EgoDex percentage -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(
    data=melted,
    x='Egodex data %',
    y='Success Rate',
    hue='Task',
    marker='o',
    ax=ax
)
ax.set_title('Success Rate vs. EgoDex Data Percentage')
ax.set_xlabel('EgoDex Data (%)')
ax.set_ylabel('Success Rate')
ax.set_xticks(sorted(df['Egodex data %'].unique()))
ax.set_ylim(0, 1.05)
ax.plot(
    df_sorted['Egodex data %'],
    df_sorted['Average Success Rate'],
    color='black',
    linestyle='--',
    marker='o',
    label='Average'
)
baseline_handles = add_baseline_lines(ax, df_sorted['Egodex data %'].values)
handles, labels = ax.get_legend_handles_labels()
handles += baseline_handles
labels += [h.get_label() for h in baseline_handles]
ax.legend(handles, labels, title='Task', bbox_to_anchor=(1.02, 1), loc='upper left')
save_plot(fig, "sr_vs_percentage.png")

# ---- Plot 2: Success Rate vs number of human videos (log scale) ------------
fig, ax = plt.subplots(figsize=(8, 6))
nonzero_videos = melted['Num videos'] > 0
sns.lineplot(
    data=melted[nonzero_videos],
    x='Num videos',
    y='Success Rate',
    hue='Task',
    marker='o',
    ax=ax
)
ax.set_xscale('log')
ax.set_title('Success Rate vs. Number of Human Videos (log scale)')
ax.set_xlabel('Number of Videos (log scale)')
ax.set_ylabel('Success Rate')
ax.set_ylim(0, 1.05)
if df_sorted['Num videos'].gt(0).any():
    ax.plot(
        df_sorted.loc[df_sorted['Num videos'] > 0, 'Num videos'],
        df_sorted.loc[df_sorted['Num videos'] > 0, 'Average Success Rate'],
        color='black',
        linestyle='--',
        marker='o',
        label='Average'
    )
baseline_handles = add_baseline_lines(ax, df_sorted['Num videos'].values)
handles, labels = ax.get_legend_handles_labels()
handles += baseline_handles
labels += [h.get_label() for h in baseline_handles]
ax.legend(handles, labels, title='Task', bbox_to_anchor=(1.02, 1), loc='upper left')
save_plot(fig, "sr_vs_num_videos.png")

# ---- Plot 3: Success Rate vs hours of human video --------------------------
fig, ax = plt.subplots(figsize=(8, 6))
nonzero_hours = melted['Hours'] > 0
sns.lineplot(
    data=melted[nonzero_hours],
    x='Hours',
    y='Success Rate',
    hue='Task',
    marker='o',
    ax=ax
)
ax.set_xscale('log')
ax.set_title('Success Rate vs. Hours of Human Video')
ax.set_xlabel('Hours of Human Video')
ax.set_ylabel('Success Rate')
ax.set_ylim(0, 1.05)
if df_sorted['Hours'].gt(0).any():
    ax.plot(
        df_sorted.loc[df_sorted['Hours'] > 0, 'Hours'],
        df_sorted.loc[df_sorted['Hours'] > 0, 'Average Success Rate'],
        color='black',
        linestyle='--',
        marker='o',
        label='Average'
    )
baseline_handles = add_baseline_lines(ax, df_sorted['Hours'].values)
handles, labels = ax.get_legend_handles_labels()
handles += baseline_handles
labels += [h.get_label() for h in baseline_handles]
ax.legend(handles, labels, title='Task', bbox_to_anchor=(1.02, 1), loc='upper left')
save_plot(fig, "sr_vs_hours.png")

# ---- Plot 4: Average scaling projection (hours only) ------------------------
fig, ax = plt.subplots(figsize=(8, 6))

hours_avg = df_sorted['Hours'].values
success_avg = df_sorted['Average Success Rate'].values

positive_mask = hours_avg > 0
hours_avg_pos = hours_avg[positive_mask]
success_avg_pos = success_avg[positive_mask]

ego_hours_curve = prepare_curve(hours_avg_pos, success_avg_pos, SCALING_TARGET_HOURS, args.fit_mode, args.remove_hours_leq_one)
robot_hours_curve = prepare_curve(robot_hours_raw, robot_success_raw, SCALING_TARGET_HOURS, args.fit_mode, args.remove_hours_leq_one)

if ego_hours_curve['scatter_x'].size > 0:
    ego_scatter_x = ego_hours_curve['scatter_x']
    ego_scatter_y = ego_hours_curve['scatter_y']
    if (
        args.remove_hours_leq_one
        and ego_hours_curve.get('scatter_used_mask') is not None
        and ego_hours_curve['scatter_used_mask'].size == ego_scatter_x.size
    ):
        mask = ego_hours_curve['scatter_used_mask']
        ego_scatter_x = ego_scatter_x[mask]
        ego_scatter_y = ego_scatter_y[mask]
    if ego_scatter_x.size > 0:
        ax.scatter(ego_scatter_x, ego_scatter_y, color='black', label='Egovideo pretraining average (observed)')
if ego_hours_curve['projection_x'].size > 0:
    fit_name = f"Egovideo pretraining {args.fit_mode} fit"
    ax.plot(ego_hours_curve['projection_x'], ego_hours_curve['projection_y'], '-', color='tab:blue', label=fit_name)

if robot_hours_curve['scatter_x'].size > 0:
    robot_scatter_x = robot_hours_curve['scatter_x']
    robot_scatter_y = robot_hours_curve['scatter_y']
    if (
        args.remove_hours_leq_one
        and robot_hours_curve.get('scatter_used_mask') is not None
        and robot_hours_curve['scatter_used_mask'].size == robot_scatter_x.size
    ):
        mask_robot = robot_hours_curve['scatter_used_mask']
        robot_scatter_x = robot_scatter_x[mask_robot]
        robot_scatter_y = robot_scatter_y[mask_robot]
    if robot_scatter_x.size > 0:
        ax.scatter(robot_scatter_x, robot_scatter_y, color='tab:orange', label='Robot data pretraining average (observed)')
if robot_hours_curve['projection_x'].size > 0:
    fit_name_robot = f"Robot data pretraining {args.fit_mode} fit"
    ax.plot(robot_hours_curve['projection_x'], robot_hours_curve['projection_y'], color='tab:red', linestyle='--', label=fit_name_robot)

ax.set_xscale('log')
ax.set_title('Average Success Rate Scaling Projection')
ax.set_xlabel('Hours of Data (log scale)')
ax.set_ylabel('Average Success Rate')
ax.set_ylim(0, 1.05)
positive_hours = ego_hours_curve['scatter_x'][ego_hours_curve['scatter_x'] > 0]
if positive_hours.size > 0:
    min_power = int(np.floor(np.log10(positive_hours.min())))
else:
    min_power = 0
max_power = int(np.ceil(np.log10(SCALING_TARGET_HOURS)))
xticks = [10 ** p for p in range(min_power, max_power + 1)]
ax.set_xticks(xticks)
ax.set_xticklabels([f"$10^{p}$" for p in range(min_power, max_power + 1)])
ax.legend(loc='lower right')

if ego_hours_curve['coeffs'] is not None:
    predicted = np.clip(evaluate_curve(ego_hours_curve['coeffs'], [SCALING_TARGET_HOURS], args.fit_mode)[0], 0.0, 1.0)
    print(f"Projected average success rate at {SCALING_TARGET_HOURS:.0e} hours: {predicted*100:.2f}%")

save_plot(fig, "sr_vs_hours_projection.png")

# ---- Plot 5: Operator hours projection --------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

operator_hours_all = hours_avg_pos * EGOVIDEO_TO_OPERATOR
robot_operator_hours_all = robot_hours_raw * ROBOTDATA_TO_OPERATOR

ego_operator_curve = prepare_curve(
    operator_hours_all,
    success_avg_pos,
    SCALING_TARGET_OPERATOR,
    args.fit_mode,
    args.remove_hours_leq_one,
    min_fit_x=EGOVIDEO_TO_OPERATOR,
)
robot_operator_curve = prepare_curve(
    robot_operator_hours_all,
    robot_success_raw,
    SCALING_TARGET_OPERATOR,
    args.fit_mode,
    args.remove_hours_leq_one,
    min_fit_x=ROBOTDATA_TO_OPERATOR,
)

if ego_operator_curve['projection_x'].size > 0:
    if ego_operator_curve['scatter_x'].size > 0:
        ego_op_scatter_x = ego_operator_curve['scatter_x']
        ego_op_scatter_y = ego_operator_curve['scatter_y']
        if (
            args.remove_hours_leq_one
            and ego_operator_curve.get('scatter_used_mask') is not None
            and ego_operator_curve['scatter_used_mask'].size == ego_op_scatter_x.size
        ):
            mask = ego_operator_curve['scatter_used_mask']
            ego_op_scatter_x = ego_op_scatter_x[mask]
            ego_op_scatter_y = ego_op_scatter_y[mask]
        if ego_op_scatter_x.size > 0:
            ax.scatter(
                ego_op_scatter_x,
                ego_op_scatter_y,
                color='tab:blue',
                alpha=0.8,
                label='_nolegend_',
            )
    ego_label = (
        f"Egovideo pretraining {args.fit_mode} fit\n"
        f"(1 hr -> {EGOVIDEO_TO_OPERATOR:.2f} operator hr)"
    )
    ax.plot(
        ego_operator_curve['projection_x'],
        ego_operator_curve['projection_y'],
        color='tab:blue',
        label=ego_label,
    )
if robot_operator_curve['projection_x'].size > 0:
    if robot_operator_curve['scatter_x'].size > 0:
        robot_op_scatter_x = robot_operator_curve['scatter_x']
        robot_op_scatter_y = robot_operator_curve['scatter_y']
        if (
            args.remove_hours_leq_one
            and robot_operator_curve.get('scatter_used_mask') is not None
            and robot_operator_curve['scatter_used_mask'].size == robot_op_scatter_x.size
        ):
            mask = robot_operator_curve['scatter_used_mask']
            robot_op_scatter_x = robot_op_scatter_x[mask]
            robot_op_scatter_y = robot_op_scatter_y[mask]
        if robot_op_scatter_x.size > 0:
            ax.scatter(
                robot_op_scatter_x,
                robot_op_scatter_y,
                color='tab:red',
                alpha=0.8,
                marker='s',
                label='_nolegend_',
            )
    robot_label = (
        f"Robot data pretraining {args.fit_mode} fit\n"
        f"(1 hr -> {ROBOTDATA_TO_OPERATOR:.2f} operator hr)"
    )
    ax.plot(
        robot_operator_curve['projection_x'],
        robot_operator_curve['projection_y'],
        color='tab:red',
        linestyle='--',
        label=robot_label,
    )

ax.set_xscale('log')
ax.set_title('Success Rate vs. Operator Hours (Projection)')
ax.set_xlabel('Operator Hours (log scale)')
ax.set_ylabel('Average Success Rate')
ax.set_ylim(0, 1.05)
if ego_operator_curve['projection_x'].size > 0:
    min_power_op = int(np.floor(np.log10(max(ego_operator_curve['projection_x'][0], 1e-12))))
else:
    min_power_op = int(np.floor(np.log10(max(operator_hours_all.min(), 1e-12))))
min_power_op = max(min_power_op, 0)
max_power_op = int(np.ceil(np.log10(SCALING_TARGET_OPERATOR)))
xticks_op = [10 ** p for p in range(min_power_op, max_power_op + 1)]
ax.set_xticks(xticks_op)
ax.set_xticklabels([f"$10^{p}$" for p in range(min_power_op, max_power_op + 1)])
# Ensure x-axis starts at 10^0 (1 operator hour)
x_axis_min = 10 ** min_power_op
curve_max_values = []
if ego_operator_curve['projection_x'].size > 0:
    curve_max_values.append(ego_operator_curve['projection_x'].max())
if robot_operator_curve['projection_x'].size > 0:
    curve_max_values.append(robot_operator_curve['projection_x'].max())
if not curve_max_values:
    curve_max_values.append(SCALING_TARGET_OPERATOR)
x_axis_max = max(curve_max_values)
ax.set_xlim(x_axis_min, x_axis_max)
ax.legend(loc='lower right')

if ego_operator_curve['coeffs'] is not None:
    predicted_operator = np.clip(evaluate_curve(ego_operator_curve['coeffs'], [SCALING_TARGET_OPERATOR], args.fit_mode)[0], 0.0, 1.0)
    print(f"Projected average success rate at {SCALING_TARGET_OPERATOR:.0e} operator hours: {predicted_operator*100:.2f}%")

if ego_operator_curve['coeffs'] is not None and ego_operator_curve['projection_x'].size > 0:
    ego_min_fit = ego_operator_curve['fit_x'].min() if ego_operator_curve['fit_x'].size > 0 else ego_operator_curve['projection_x'][0]
    for target_sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        if not (0 < target_sr < 1):
            continue
        hours_needed = invert_curve(ego_operator_curve['coeffs'], target_sr, args.fit_mode)
        if not np.isfinite(hours_needed):
            continue
        hours_needed = max(hours_needed, ego_min_fit)
        ax.scatter(target_sr, hours_needed, color='tab:blue')
        ax.text(target_sr, hours_needed, f"{hours_needed:.1e}", fontsize=10, ha='left', va='bottom', color='tab:blue')
        if robot_operator_curve['coeffs'] is not None and robot_operator_curve['projection_x'].size > 0:
            robot_min_fit = robot_operator_curve['fit_x'].min() if robot_operator_curve['fit_x'].size > 0 else robot_operator_curve['projection_x'][0]
            robot_hours_needed = invert_curve(robot_operator_curve['coeffs'], target_sr, args.fit_mode)
            if np.isfinite(robot_hours_needed):
                robot_hours_needed = max(robot_hours_needed, robot_min_fit)
                ax.scatter(target_sr, robot_hours_needed, color='tab:red')
                ax.text(target_sr, robot_hours_needed, f"{robot_hours_needed:.1e}", fontsize=10, ha='left', va='top', color='tab:red')

save_plot(fig, "sr_vs_operator_hours_projection.png")

# ---- Plot 6: Total cost vs target success rate ------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

if ego_operator_curve['projection_x'].size > 0:
    target_hours = ego_operator_curve['projection_x']
    if robot_operator_curve['projection_x'].size > 0:
        target_hours = np.union1d(target_hours, robot_operator_curve['projection_x'])
    target_hours = np.union1d(target_hours, [SCALING_TARGET_OPERATOR])

    ego_hours_for_one = invert_curve(ego_operator_curve['coeffs'], 1.0, args.fit_mode)
    if np.isfinite(ego_hours_for_one) and ego_hours_for_one > 0:
        target_hours = np.union1d(target_hours, [ego_hours_for_one])

    if robot_operator_curve['coeffs'] is not None:
        robot_hours_for_one = invert_curve(robot_operator_curve['coeffs'], 1.0, args.fit_mode)
        if np.isfinite(robot_hours_for_one) and robot_hours_for_one > 0:
            target_hours = np.union1d(target_hours, [robot_hours_for_one])

    ego_cost = target_hours * OPERATOR_HOUR_COST
    ego_success_curve = np.clip(evaluate_curve(ego_operator_curve['coeffs'], target_hours, args.fit_mode), 0.0, 1.0)
    ax.plot(ego_success_curve, ego_cost, color='tab:blue', label='Egovideo pretraining cost curve')

    if robot_operator_curve['projection_x'].size > 0 and robot_operator_curve['coeffs'] is not None:
        robot_cost = target_hours * OPERATOR_HOUR_COST
        robot_success_curve = np.clip(evaluate_curve(robot_operator_curve['coeffs'], target_hours, args.fit_mode), 0.0, 1.0)
        ax.plot(robot_success_curve, robot_cost, color='tab:red', linestyle='--', label='Robot data pretraining cost curve')

    for target_sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        if not (0 < target_sr < 1):
            continue
        ego_hours_needed = invert_curve(ego_operator_curve['coeffs'], target_sr, args.fit_mode)
        if not np.isfinite(ego_hours_needed):
            continue
        ego_hours_needed = max(ego_hours_needed, ego_operator_curve['fit_x'].min()) if ego_operator_curve['fit_x'].size > 0 else ego_hours_needed
        ego_cost_needed = ego_hours_needed * OPERATOR_HOUR_COST
        ego_success_actual = np.clip(
            evaluate_curve(ego_operator_curve['coeffs'], [ego_hours_needed], args.fit_mode)[0],
            0.0,
            1.0,
        )
        ax.scatter(ego_success_actual, ego_cost_needed, color='tab:blue')
        if ego_cost_needed < 1e3:
            label = f"${ego_cost_needed:.0f}"
        elif ego_cost_needed < 9e4:
            label = f"${ego_cost_needed/1e3:.0f} k"
        else:
            label = f"${ego_cost_needed/1e6:.1f} M"
        ax.text(
            ego_success_actual,
            ego_cost_needed,
            label,
            fontsize=10,
            ha='left',
            va='bottom',
            color='tab:blue',
        )

        if robot_operator_curve['coeffs'] is not None and robot_operator_curve['projection_x'].size > 0:
            robot_hours_needed = invert_curve(robot_operator_curve['coeffs'], target_sr, args.fit_mode)
            if np.isfinite(robot_hours_needed):
                robot_hours_needed = max(robot_hours_needed, robot_operator_curve['fit_x'].min()) if robot_operator_curve['fit_x'].size > 0 else robot_hours_needed
                robot_cost_needed = robot_hours_needed * OPERATOR_HOUR_COST
                robot_success_actual = np.clip(
                    evaluate_curve(robot_operator_curve['coeffs'], [robot_hours_needed], args.fit_mode)[0],
                    0.0,
                    1.0,
                )
                ax.scatter(robot_success_actual, robot_cost_needed, color='tab:red')
                if robot_cost_needed < 1e3:
                    label_robot = f"${robot_cost_needed:.0f}"
                elif robot_cost_needed < 9e4:
                    label_robot = f"${robot_cost_needed/1e3:.0f} k"
                else:
                    label_robot = f"${robot_cost_needed/1e6:.1f} M"
                ax.text(
                    robot_success_actual,
                    robot_cost_needed,
                    label_robot,
                    fontsize=10,
                    ha='left',
                    va='top',
                    color='tab:red',
                )
else:
    ax.text(0.5, 0.5, "Insufficient data for cost projection", transform=ax.transAxes, ha='center', va='center')

ax.set_title('Total Operator Cost vs. Target Success Rate')
ax.set_xlabel('Success Rate')
ax.set_ylabel('Total Operator Cost (USD) - log scale')
ax.set_xlim(0, 1.0)
ax.set_yscale('log')
ax.legend(loc='upper left')
ax.axvline(0.6, color='gray', linestyle=':', linewidth=1.2)
ax.text(0.6, ax.get_ylim()[0], "Common pretraining target", rotation=90, va='bottom', ha='right', color='gray')

save_plot(fig, "cost_vs_target_success.png")
