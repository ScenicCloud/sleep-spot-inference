#!/usr/bin/env python3
"""生成论文插图（图 1–图 3）。

用法：python3 src/make_figures.py
输出：paper/figures/fig1_lifecycle.png
      paper/figures/fig2_recovery_time.png
      paper/figures/fig3_memory.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

EDGE = "#2b2b2b"
FILL = "#eaf1f7"
ACCENT = "#2f5d8a"
MUTED = "#9fb3c8"


def fig1_lifecycle():
    """图 1：SleepSpot 推理实例的生命周期状态迁移。"""
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    boxes = {
        "冷启动": (0.03, 0.62, 0.23, 0.25),
        "Serving": (0.39, 0.62, 0.23, 0.25),
        "Sleeping": (0.74, 0.62, 0.23, 0.25),
        "Finalizing": (0.39, 0.09, 0.23, 0.25),
    }
    for name, (x, y, w, h) in boxes.items():
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.035",
                linewidth=1.3, edgecolor=EDGE, facecolor=FILL,
            )
        )
        ax.text(x + w / 2, y + h / 2, name, ha="center", va="center", fontsize=13, color=EDGE)

    def arrow(xy, xytext, rad=0.0):
        ax.annotate(
            "", xy=xy, xytext=xytext,
            arrowprops=dict(
                arrowstyle="-|>", color=ACCENT, lw=1.5,
                connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
            ),
        )

    # 冷启动 → Serving
    arrow((0.39, 0.745), (0.26, 0.745))
    ax.text(0.325, 0.80, "就绪", ha="center", va="bottom", fontsize=9.5, color=EDGE)
    # Serving ⇄ Sleeping
    arrow((0.74, 0.80), (0.62, 0.80))
    ax.text(0.68, 0.885, "POST /sleep?level=1", ha="center", va="bottom", fontsize=9, color=EDGE)
    arrow((0.62, 0.685), (0.74, 0.685))
    ax.text(0.68, 0.605, "POST /wake_up", ha="center", va="top", fontsize=9, color=EDGE)
    # Serving → Finalizing
    arrow((0.505, 0.34), (0.505, 0.62))
    ax.text(0.535, 0.48, "回收发生", ha="left", va="center", fontsize=9.5, color=EDGE)
    # Finalizing → 冷启动（重建）
    arrow((0.155, 0.62), (0.43, 0.215), rad=0.32)
    ax.text(0.20, 0.33, "新实例重建", ha="center", va="center", fontsize=9.5, color=EDGE)
    ax.text(0.20, 0.275, "（冷启动基线）", ha="center", va="center", fontsize=9.5, color=EDGE)

    fig.savefig(os.path.join(OUT, "fig1_lifecycle.png"))
    plt.close(fig)


def fig2_recovery():
    """图 2：两个模型规模下端到端恢复时间对比（对数纵轴）。"""
    labels = ["冷启动", "Sleep L1 唤醒", "Sleep L2 唤醒"]
    v15 = np.array([50.04, 0.21, 0.08])
    v7 = np.array([54.53, 0.82, 0.18])
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    b1 = ax.bar(x - w / 2, v15, w, label="Qwen2.5-1.5B", color=MUTED,
                edgecolor=EDGE, linewidth=0.6)
    b2 = ax.bar(x + w / 2, v7, w, label="Qwen2.5-7B", color=ACCENT,
                edgecolor=EDGE, linewidth=0.6)

    ax.set_yscale("log")
    ax.set_ylim(0.04, 400)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("端到端恢复时间（秒，对数刻度）", fontsize=11)
    ax.grid(axis="y", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    for bars in (b1, b2):
        for b in bars:
            ax.annotate(
                f"{b.get_height():g}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=9, color=EDGE,
            )
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    fig.savefig(os.path.join(OUT, "fig2_recovery_time.png"))
    plt.close(fig)


def fig3_memory():
    """图 3：睡眠前后 GPU 显存占用（1.5B 睡眠前为实测，7B 睡眠前为估算）。"""
    names = [
        "Qwen2.5-1.5B\n（21 012 → 980 MiB）",
        "Qwen2.5-7B\n（约 20 GiB → 1 058 MiB）",
    ]
    retained = np.array([980.0, 1058.0])      # MiB，睡眠后实测
    total = np.array([21012.0, 20490.0])      # MiB；1.5B 实测，7B 估算（约 20 GiB）
    freed = total - retained
    pct_freed = freed / total * 100
    notes = [
        f"释放 {freed[0]:,.0f} MiB（{pct_freed[0]:.1f}%）",
        f"释放约 19 GiB（≈{pct_freed[1]:.0f}%）",
    ]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    ax.barh(y, pct_freed, color=ACCENT, edgecolor=EDGE, linewidth=0.6)
    ax.barh(y, 100 - pct_freed, left=pct_freed, color="#dfe8f0",
            edgecolor=EDGE, linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("占睡眠前 GPU 显存的比例（%）", fontsize=11)
    ax.grid(axis="x", linestyle=":", color="#cccccc", linewidth=0.6)
    ax.set_axisbelow(True)

    for i in range(len(names)):
        ax.text(pct_freed[i] / 2, i, notes[i],
                ha="center", va="center", fontsize=9.5, color="white")

    fig.savefig(os.path.join(OUT, "fig3_memory.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig1_lifecycle()
    fig2_recovery()
    fig3_memory()
    print("figures written to", os.path.normpath(OUT))
