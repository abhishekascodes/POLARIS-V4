#!/usr/bin/env python3
"""Generate Bloomberg-style GRPO training results chart (light theme)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Actual data from training_results.json (Qwen2.5-3B-Instruct) ──
baseline_rewards = [10.87, 11.36, 14.98, 12.29, 17.35]
post_rewards = [28.37, 33.21, 25.18, 35.67, 28.82]
baseline_avg = 13.37
post_avg = 30.25
improvement_pct = 126.3

baseline_tom = {"tom_accuracy": 0.01, "tom_total": 85, "coalitions": 12}
post_tom = {"tom_accuracy": 0.01, "tom_total": 107, "coalitions": 35}

curriculum = [
    {"label": "Easy",    "avg_reward": 40.80, "survivals": 3, "total": 3},
    {"label": "Medium",  "avg_reward": 38.30, "survivals": 2, "total": 3},
    {"label": "Hard",    "avg_reward": 24.90, "survivals": 0, "total": 3},
    {"label": "Extreme", "avg_reward": 22.70, "survivals": 0, "total": 3},
]

# ── Bloomberg-inspired palette ──
BG         = '#FFFFFF'
PANEL_BG   = '#F7F7F8'
GRID_COLOR = '#E0E0E0'
TEXT_DARK   = '#1A1A2E'
TEXT_MED    = '#555568'
TEXT_LIGHT  = '#8E8EA0'
BLUE_PRI   = '#2962FF'   # Bloomberg blue
BLUE_SEC   = '#BBDEFB'
ORANGE     = '#FF6D00'
GREEN_UP   = '#00C853'
RED_DN     = '#D50000'
TEAL       = '#00897B'
PURPLE     = '#7C4DFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial'],
    'text.color': TEXT_DARK,
    'axes.labelcolor': TEXT_MED,
    'xtick.color': TEXT_MED,
    'ytick.color': TEXT_MED,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

fig = plt.figure(figsize=(16, 10.5), facecolor=BG)

# ── Header ──
fig.text(0.04, 0.97, 'POLARIS v3', fontsize=22, fontweight='bold',
         color=BLUE_PRI, va='top')
fig.text(0.175, 0.97, 'GRPO Training Results', fontsize=22, fontweight='normal',
         color=TEXT_DARK, va='top')
fig.text(0.04, 0.942,
         'Qwen2.5-3B-Instruct  |  4-bit QLoRA  |  RTX 5080  |  100 steps  |  788s training time',
         fontsize=10, color=TEXT_LIGHT, va='top')
# Thin blue line under header
line = plt.Line2D([0.03, 0.97], [0.93, 0.93], transform=fig.transFigure,
                  color=BLUE_PRI, linewidth=2.5)
fig.add_artist(line)

# ════════════════════════════════════════════════
# Panel 1: Reward Improvement (top-left)
# ════════════════════════════════════════════════
ax1 = fig.add_axes([0.06, 0.52, 0.40, 0.36])
ax1.set_facecolor(PANEL_BG)

bars1 = ax1.bar(['Baseline\n(Pre-Training)', 'Post-GRPO\n(Trained)'],
                [baseline_avg, post_avg],
                color=[TEXT_LIGHT, BLUE_PRI], width=0.45, zorder=3,
                edgecolor='none')

# Value labels
ax1.text(0, baseline_avg + 1.2, f'{baseline_avg:.2f}',
         ha='center', fontsize=15, fontweight='bold', color=TEXT_MED)
ax1.text(1, post_avg + 1.2, f'{post_avg:.2f}',
         ha='center', fontsize=15, fontweight='bold', color=BLUE_PRI)

# Improvement badge
ax1.annotate(f'+{improvement_pct:.1f}%',
             xy=(0.5, 0.88), xycoords='axes fraction', ha='center',
             fontsize=20, fontweight='bold', color=GREEN_UP,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F5E9',
                       edgecolor=GREEN_UP, linewidth=1.5))

# Arrow between bars
ax1.annotate('', xy=(0.92, post_avg * 0.7), xytext=(0.08, baseline_avg * 0.7),
             arrowprops=dict(arrowstyle='->', color=GREEN_UP, lw=2))

ax1.set_ylabel('Avg Episode Reward')
ax1.set_title('Reward Improvement', fontweight='bold', color=TEXT_DARK, pad=14, loc='left')
ax1.set_ylim(0, 40)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax1.grid(axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
ax1.set_axisbelow(True)
for spine in ax1.spines.values(): spine.set_visible(False)
ax1.tick_params(length=0)

# ════════════════════════════════════════════════
# Panel 2: Per-Episode Comparison (top-right)
# ════════════════════════════════════════════════
ax2 = fig.add_axes([0.55, 0.52, 0.40, 0.37])
ax2.set_facecolor(PANEL_BG)

x = np.arange(1, 6)
w = 0.30
ax2.bar(x - w/2, baseline_rewards, w, color=TEXT_LIGHT, label='Baseline',
        zorder=3, edgecolor='none')
ax2.bar(x + w/2, post_rewards, w, color=BLUE_PRI, label='Post-GRPO',
        zorder=3, edgecolor='none')

# Value labels
for i, (b, p) in enumerate(zip(baseline_rewards, post_rewards)):
    ax2.text(x[i] - w/2, b + 0.7, f'{b:.1f}', ha='center', fontsize=7.5,
             color=TEXT_LIGHT, fontweight='bold')
    ax2.text(x[i] + w/2, p + 0.7, f'{p:.1f}', ha='center', fontsize=7.5,
             color=BLUE_PRI, fontweight='bold')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Total Reward')
ax2.set_title('Per-Episode Breakdown', fontweight='bold', color=TEXT_DARK, pad=14, loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Ep {i}' for i in x])
ax2.set_ylim(0, 42)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax2.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR, fontsize=9,
           loc='upper left', ncol=2)
ax2.grid(axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
ax2.set_axisbelow(True)
for spine in ax2.spines.values(): spine.set_visible(False)
ax2.tick_params(length=0)

# ════════════════════════════════════════════════
# Panel 3: Curriculum Escalation (bottom-left)
# ════════════════════════════════════════════════
ax3 = fig.add_axes([0.06, 0.06, 0.40, 0.37])
ax3.set_facecolor(PANEL_BG)

labels3 = [c['label'] for c in curriculum]
rewards3 = [c['avg_reward'] for c in curriculum]
colors3 = [GREEN_UP, ORANGE, RED_DN, PURPLE]

bars3 = ax3.bar(labels3, rewards3, color=colors3, width=0.50, zorder=3, edgecolor='none')

for i, (bar, c) in enumerate(zip(bars3, curriculum)):
    r = c['avg_reward']
    ax3.text(bar.get_x() + bar.get_width()/2, r + 1.2,
             f'{r:.1f}', ha='center', fontweight='bold', fontsize=12, color=TEXT_DARK)
    surv_text = f"{c['survivals']}/{c['total']} survived"
    surv_color = GREEN_UP if c['survivals'] > 0 else RED_DN
    ax3.text(bar.get_x() + bar.get_width()/2, -2.8,
             surv_text, ha='center', fontsize=8, color=surv_color, fontweight='bold')

ax3.set_ylabel('Avg Reward')
ax3.set_title('Curriculum Escalation (Post-Training)', fontweight='bold',
              color=TEXT_DARK, pad=14, loc='left')
ax3.set_ylim(-5, 50)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax3.grid(axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
ax3.set_axisbelow(True)
for spine in ax3.spines.values(): spine.set_visible(False)
ax3.tick_params(length=0)

# ════════════════════════════════════════════════
# Panel 4: Coalition & Behavioral Metrics (bottom-right)
# ════════════════════════════════════════════════
ax4 = fig.add_axes([0.55, 0.06, 0.40, 0.37])
ax4.set_facecolor(PANEL_BG)

# Coalition formation (count) and Survival rate (%)
metric_names = ['Coalition\nFormation', 'Survival\nRate']
before_vals = [12, 0]
after_vals  = [35, 20]
change_labels = ['2.9×', '0% → 20%']

x4 = np.arange(len(metric_names))
w4 = 0.30
b4a = ax4.bar(x4 - w4/2, before_vals, w4, color=TEXT_LIGHT, label='Baseline',
              zorder=3, edgecolor='none')
b4b = ax4.bar(x4 + w4/2, after_vals, w4, color=BLUE_PRI, label='Post-GRPO',
              zorder=3, edgecolor='none')

# Value labels
for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
    bv_label = str(bv) if i == 0 else f'{bv}%'
    av_label = str(av) if i == 0 else f'{av}%'
    ax4.text(x4[i] - w4/2, bv + 1.2, bv_label, ha='center', fontsize=11,
             color=TEXT_MED, fontweight='bold')
    ax4.text(x4[i] + w4/2, av + 1.2, av_label, ha='center', fontsize=11,
             color=BLUE_PRI, fontweight='bold')

# Change annotations
for i, cl in enumerate(change_labels):
    ax4.text(x4[i], max(after_vals[i], before_vals[i]) + 7, cl,
             ha='center', fontsize=11, fontweight='bold', color=GREEN_UP,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                       edgecolor='none'))

ax4.set_ylabel('Count / %')
ax4.set_title('Agent Behavior Metrics', fontweight='bold', color=TEXT_DARK, pad=14, loc='left')
ax4.set_xticks(x4)
ax4.set_xticklabels(metric_names)
ax4.set_ylim(0, 55)
ax4.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax4.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR, fontsize=9,
           loc='upper right', ncol=2)
ax4.grid(axis='y', color=GRID_COLOR, linewidth=0.7, zorder=0)
ax4.set_axisbelow(True)
for spine in ax4.spines.values(): spine.set_visible(False)
ax4.tick_params(length=0)

# ── Footer ──
fig.text(0.04, 0.01,
         'Source: POLARIS v3 Governance Engine  |  Training: GRPO w/ TRL  |  Hardware: NVIDIA RTX 5080 16GB',
         fontsize=8, color=TEXT_LIGHT, va='bottom')
fig.text(0.97, 0.01, 'openenv-polaris', fontsize=8, color=BLUE_PRI,
         va='bottom', ha='right', fontweight='bold')

# ── Save ──
save_path = r"c:\Users\AbhishekPC\Desktop\OpenENV Hackathon\OpenENV Hackathon\openenv\outputs\grpo_training\grpo_3b_training_results.png"
plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG, pad_inches=0.3)
plt.close()
print(f"Chart saved: {save_path}")
