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
    help="Exclude samples with hours <= 1 when fitting power-law curves (EgoVideo and Robot data).",
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
BASELINE_STYLES = {
    "π0": {"color": "dimgray", "linestyle": "--"},
    "DP": {"color": "saddlebrown", "linestyle": ":"},
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
sns.set(style="whitegrid", font_scale=1.2)

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

fit_mask_ego = hours_avg_pos > 1 if args.remove_hours_leq_one else np.ones_like(hours_avg_pos, dtype=bool)
if fit_mask_ego.sum() >= 2:
    hours_fit = hours_avg_pos[fit_mask_ego]
    success_fit = success_avg_pos[fit_mask_ego]
else:
    hours_fit = hours_avg_pos
    success_fit = success_avg_pos

log_hours = np.log10(hours_fit)
coeffs = np.polyfit(log_hours, success_fit, deg=1)

projection_hours = np.logspace(
    np.log10(hours_fit.min()),
    np.log10(SCALING_TARGET_HOURS),
    num=400
)
projection_success = np.polyval(coeffs, np.log10(projection_hours))
projection_success = np.clip(projection_success, 0.0, 1.0)

avg_label_obs = 'Egovideo pretraining average (observed)'
avg_label_fit = 'Egovideo pretraining power-law fit'
robot_label_obs = 'Robot data pretraining average (observed)'
robot_label_fit = 'Robot data pretraining power-law fit'

ax.scatter(hours_avg_pos, success_avg_pos, color='black', label=avg_label_obs)
ax.plot(projection_hours, projection_success, '-', color='tab:blue', label=avg_label_fit)

robot_positive = robot_df['Robot Data Hours'] > 0
robot_hours_all = robot_df.loc[robot_positive, 'Robot Data Hours'].values
robot_success_all = robot_df.loc[robot_positive, 'Success Rate'].values
fit_mask_robot = robot_hours_all > 1 if args.remove_hours_leq_one else np.ones_like(robot_hours_all, dtype=bool)
robot_hours_fit = robot_hours_all[fit_mask_robot]
robot_success_fit = robot_success_all[fit_mask_robot]

if robot_hours_all.size > 0:
    ax.scatter(
        robot_hours_all,
        robot_success_all,
        color='tab:orange',
        label=robot_label_obs
    )

robot_projection_hours = None
robot_projection_success = None
if robot_hours_fit.size >= 2:
    robot_coeffs = np.polyfit(np.log10(robot_hours_fit), robot_success_fit, deg=1)
    robot_projection_hours = np.logspace(
        np.log10(robot_hours_fit.min()),
        np.log10(SCALING_TARGET_HOURS),
        num=400
    )
    robot_projection_success = np.polyval(robot_coeffs, np.log10(robot_projection_hours))
    robot_projection_success = np.clip(robot_projection_success, 0.0, 1.0)
    ax.plot(
        robot_projection_hours,
        robot_projection_success,
        color='tab:red',
        linestyle='--',
        label=robot_label_fit
    )

ax.set_xscale('log')
ax.set_title('Average Success Rate Scaling Projection')
ax.set_xlabel('Hours of Human Video (log scale)')
ax.set_ylabel('Average Success Rate')
ax.set_ylim(0, 1.05)
min_power = int(np.floor(np.log10(hours_fit.min())))
max_power = int(np.ceil(np.log10(SCALING_TARGET_HOURS)))
xticks = [10 ** p for p in range(min_power, max_power + 1)]
ax.set_xticks(xticks)
ax.set_xticklabels([f"$10^{p}$" for p in range(min_power, max_power + 1)])
ax.legend(loc='lower right')

predicted = np.polyval(coeffs, np.log10(SCALING_TARGET_HOURS))
predicted = max(0.0, min(1.0, predicted))
print(f"Projected average success rate at {SCALING_TARGET_HOURS:.0e} hours: {predicted*100:.2f}%")

save_plot(fig, "sr_vs_hours_projection.png")

# ---- Plot 5: Operator hours projection --------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

operator_hours_all = hours_avg_pos * EGOVIDEO_TO_OPERATOR
if fit_mask_ego.sum() >= 2:
    operator_hours_fit = operator_hours_all[fit_mask_ego]
    success_fit_op = success_fit
else:
    operator_hours_fit = operator_hours_all
    success_fit_op = success_avg_pos

log_operator = np.log10(operator_hours_fit)
coeffs_operator = np.polyfit(log_operator, success_fit_op, deg=1)

SCALING_TARGET_OPERATOR = SCALING_TARGET_HOURS * EGOVIDEO_TO_OPERATOR
projection_operator_hours = np.logspace(
    np.log10(operator_hours_fit.min()),
    np.log10(SCALING_TARGET_OPERATOR),
    num=400
)
projection_operator_success = np.polyval(coeffs_operator, np.log10(projection_operator_hours))
projection_operator_success = np.clip(projection_operator_success, 0.0, 1.0)

ax.plot(
    projection_operator_hours,
    projection_operator_success,
    color='tab:blue',
    label='Egovideo pretraining power-law fit'
)

robot_operator_hours = robot_hours_fit * ROBOTDATA_TO_OPERATOR

robot_success_op = robot_success_fit
if robot_operator_hours.size >= 2:
    robot_coeffs_op = np.polyfit(np.log10(robot_operator_hours), robot_success_op, deg=1)
    robot_projection_operator_hours = np.logspace(
        np.log10(robot_operator_hours.min()),
        np.log10(SCALING_TARGET_OPERATOR),
        num=400
    )
    robot_projection_operator_success = np.polyval(robot_coeffs_op, np.log10(robot_projection_operator_hours))
    robot_projection_operator_success = np.clip(robot_projection_operator_success, 0.0, 1.0)
    ax.plot(
        robot_projection_operator_hours,
        robot_projection_operator_success,
        color='tab:red',
        linestyle='--',
        label='Robot data pretraining power-law fit'
    )

ax.set_xscale('log')
ax.set_title('Success Rate vs. Operator Hours (Projection)')
ax.set_xlabel('Operator Hours (log scale)')
ax.set_ylabel('Average Success Rate')
ax.set_ylim(0, 1.05)

min_power_op = int(np.floor(np.log10(operator_hours_fit.min())))
max_power_op = int(np.ceil(np.log10(SCALING_TARGET_OPERATOR)))
xticks_op = [10 ** p for p in range(min_power_op, max_power_op + 1)]
ax.set_xticks(xticks_op)
ax.set_xticklabels([f"$10^{p}$" for p in range(min_power_op, max_power_op + 1)])
ax.legend(loc='lower right')

predicted_operator = np.polyval(coeffs_operator, np.log10(SCALING_TARGET_OPERATOR))
predicted_operator = max(0.0, min(1.0, predicted_operator))
print(f"Projected average success rate at {SCALING_TARGET_OPERATOR:.0e} operator hours: {predicted_operator*100:.2f}%")

for target_sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
    if target_sr <= 0 or target_sr >= 1:
        continue
    hours_needed = 10 ** np.interp(
        target_sr,
        np.log10(projection_operator_success[::-1]),
        np.log10(projection_operator_hours[::-1])
    )
    hours_needed = max(hours_needed, operator_hours_fit.min())
    ax.scatter(target_sr, hours_needed, color='tab:blue')
    ax.text(target_sr, hours_needed, f"{hours_needed:.1e}", fontsize=10, ha='left', va='bottom', color='tab:blue')
    if robot_operator_hours.size >= 2:
        robot_hours_needed = 10 ** np.interp(
            target_sr,
            np.log10(robot_projection_operator_success[::-1]),
            np.log10(robot_projection_operator_hours[::-1])
        )
        robot_hours_needed = max(robot_hours_needed, robot_operator_hours.min())
        ax.scatter(target_sr, robot_hours_needed, color='tab:red')
        ax.text(target_sr, robot_hours_needed, f"{robot_hours_needed:.1e}", fontsize=10, ha='left', va='top', color='tab:red')

save_plot(fig, "sr_vs_operator_hours_projection.png")

# ---- Plot 6: Total cost vs target success rate ------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

target_hours = projection_operator_hours
if 'robot_projection_operator_hours' in locals():
    target_hours = np.union1d(target_hours, robot_projection_operator_hours)

ego_cost = target_hours * OPERATOR_HOUR_COST
ego_success_curve = np.polyval(coeffs_operator, np.log10(target_hours))
ego_success_curve = np.clip(ego_success_curve, 0.0, 1.0)

ax.plot(
    ego_success_curve,
    ego_cost,
    color='tab:blue',
    label='Egovideo pretraining cost curve'
)
if 'robot_projection_operator_hours' in locals():
    robot_cost = target_hours * OPERATOR_HOUR_COST
    robot_success_curve = np.polyval(robot_coeffs_op, np.log10(target_hours))
    robot_success_curve = np.clip(robot_success_curve, 0.0, 1.0)
    ax.plot(
        robot_success_curve,
        robot_cost,
        color='tab:red',
        linestyle='--',
        label='Robot data pretraining cost curve'
    )

ax.set_title('Total Operator Cost vs. Target Success Rate')
ax.set_xlabel('Success Rate')
ax.set_ylabel('Total Operator Cost (USD)')
ax.set_xlim(0, 1.0)
ax.set_yscale('log')
ax.legend(loc='upper left')

for target_sr in [0.4, 0.5, 0.6, 0.7, 0.8]:
    if target_sr <= 0 or target_sr >= 1:
        continue
    idx = np.argmin(np.abs(ego_success_curve - target_sr))
    cost_needed = ego_cost[idx]
    ax.scatter(ego_success_curve[idx], cost_needed, color='tab:blue')
    if cost_needed < 1e3:
        label = f"${cost_needed:.0f}"
    elif cost_needed < 9e4:
        label = f"${cost_needed/1e3:.0f} k"
    else:
        label = f"${cost_needed/1e6:.1f} M"
    ax.text(ego_success_curve[idx], cost_needed, label, fontsize=10, ha='left', va='bottom', color='tab:blue')
    if 'robot_projection_operator_hours' in locals():
        idx_robot = np.argmin(np.abs(robot_success_curve - target_sr))
        robot_cost_needed = robot_cost[idx_robot]
        ax.scatter(robot_success_curve[idx_robot], robot_cost_needed, color='tab:red')
        if robot_cost_needed < 1e3:
            label_robot = f"${robot_cost_needed:.0f}"
        elif robot_cost_needed < 9e4:
            label_robot = f"${robot_cost_needed/1e3:.0f} k"
        else:
            label_robot = f"${robot_cost_needed/1e6:.1f} M"
        ax.text(robot_success_curve[idx_robot], robot_cost_needed, label_robot, fontsize=10, ha='left', va='top', color='tab:red')

ax.axvline(0.6, color='gray', linestyle=':', linewidth=1.2)
ax.text(0.6, ax.get_ylim()[0], "Common pretraining target", rotation=90, va='bottom', ha='right', color='gray')

save_plot(fig, "cost_vs_target_success.png")
