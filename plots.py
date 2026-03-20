import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SAVE_DIR = "saved"
METRICS  = os.path.join(SAVE_DIR, "metrics.json")
DPI      = 150

COLORS = {
    "VanillaLSTM_full":  "#2196F3",
    "VanillaLSTM_short": "#F44336",
    "EBM_LSTM":          "#4CAF50",
}
LABELS = {
    "VanillaLSTM_full":  "LSTM\n(full window)",
    "VanillaLSTM_short": "LSTM\n(short window)",
    "EBM_LSTM":          "EBM-LSTM",
}
LABELS_SHORT = {
    "VanillaLSTM_full":  "LSTM (full)",
    "VanillaLSTM_short": "LSTM (short)",
    "EBM_LSTM":          "EBM-LSTM",
}
MODEL_ORDER = ["VanillaLSTM_full", "VanillaLSTM_short", "EBM_LSTM"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
    "figure.dpi":        DPI,
    "axes.spines.right": False,
    "axes.spines.top":   False,
})


def savefig(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def bar_labels(ax, bars, fmt="{:.3f}", offset_frac=0.01, max_val=None):
    if max_val is None:
        max_val = max(b.get_height() for b in bars)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + max_val * offset_frac,
                fmt.format(h), ha="center", va="bottom", fontsize=8)


def fig_core_comparison(m):
    core = m["core_comparison"]
    cfg  = m["config"]

    names  = MODEL_ORDER
    accs   = [core[n]["accuracy"]          for n in names]
    lats   = [core[n]["latency"]["mean"]   for n in names]
    params = [core[n]["params"]            for n in names]
    colors = [COLORS[n]                    for n in names]
    xlbls  = [LABELS[n]                    for n in names]
    x      = range(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"EBM-LSTM vs Vanilla LSTM  |  seq={cfg['SEQ_LEN']}, "
        f"buf={cfg['BUFFER_LEN']}, lstm_win={cfg['LSTM_WIN']}, "
        f"slots={cfg['N_SLOTS']}  |  marker anywhere in [0,T)",
        fontsize=10,
    )

    ax = axes[0]
    bars = ax.bar(x, accs, color=colors, edgecolor="black", width=0.55)
    ax.set_xticks(list(x)); ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Test Accuracy")
    ax.set_title("Classification Accuracy")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="random baseline")
    ax.legend(fontsize=8)
    bar_labels(ax, bars, "{:.3f}")

    ax = axes[1]
    bars = ax.bar(x, lats, color=colors, edgecolor="black", width=0.55)
    ax.set_xticks(list(x)); ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel("Latency (µs / batch)"); ax.set_title("Inference Latency")
    bar_labels(ax, bars, "{:.0f}")
    fl, el = lats[0], lats[2]
    ax.annotate(
        f"{fl/el:.2f}× faster",
        xy=(2, el), xytext=(1.5, (fl + el) / 2),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        fontsize=9, ha="center",
    )

    ax = axes[2]
    bars = ax.bar(x, [p/1000 for p in params], color=colors,
                  edgecolor="black", width=0.55)
    ax.set_xticks(list(x)); ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel("Parameters (thousands)"); ax.set_title("Parameter Count")
    bar_labels(ax, bars, "{:.1f}", max_val=max(params)/1000)

    plt.tight_layout()
    savefig(fig, "fig1_core_comparison")


def fig_tradeoff_scatter(m):
    core   = m["core_comparison"]
    names  = MODEL_ORDER
    accs   = [core[n]["accuracy"]        for n in names]
    lats   = [core[n]["latency"]["mean"] for n in names]
    params = [core[n]["params"]          for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Accuracy–Efficiency Tradeoff", fontsize=12)

    for i, n in enumerate(names):
        ax.scatter(lats[i], accs[i],
                   color=COLORS[n], s=params[i] / 30,
                   edgecolors="black", linewidths=0.6,
                   zorder=4, label=LABELS_SHORT[n])
        ax.annotate(
            LABELS_SHORT[n],
            xy=(lats[i], accs[i]),
            xytext=(lats[i], accs[i] + 0.012),
            fontsize=8, ha="center",
        )

    ax.set_xlabel("Inference Latency (µs / batch)")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.4, 1.1)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="random baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.text(0.02, 0.03, "Bubble area ∝ parameter count",
            transform=ax.transAxes, fontsize=8, color="gray")

    plt.tight_layout()
    savefig(fig, "fig2_tradeoff_scatter")


def fig_training_curves(m):
    curves = m["training_curves"]
    names  = MODEL_ORDER
    epochs = range(1, len(curves[names[0]]["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Training Dynamics", fontsize=12)

    ax = axes[0]
    for n in names:
        ax.plot(epochs, curves[n]["train_loss"],
                color=COLORS[n], label=LABELS_SHORT[n], lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss (CE)")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.25)

    ax = axes[1]
    for n in names:
        ax.plot(epochs, curves[n]["train_acc"],
                color=COLORS[n], lw=1.8, ls="--", alpha=0.55)
        ax.plot(epochs, curves[n]["val_acc"],
                color=COLORS[n], lw=1.8, label=LABELS_SHORT[n])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy  (solid = val, dashed = train)")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.legend(); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    savefig(fig, "fig3_training_curves")


def fig_latency_vs_seqlen(m):
    scaling = m["latency_scaling_seq"]
    Ts      = sorted(int(k) for k in scaling)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Inference Latency vs Sequence Length", fontsize=12)

    for n in MODEL_ORDER:
        means = [scaling[str(T)][n]["mean"] for T in Ts]
        p95   = [scaling[str(T)][n]["p95"]  for T in Ts]
        p50   = [scaling[str(T)][n]["p50"]  for T in Ts]
        ax.plot(Ts, means, color=COLORS[n], lw=2, marker="o",
                label=LABELS_SHORT[n])
        ax.fill_between(Ts, p50, p95, color=COLORS[n], alpha=0.12)

    ax.set_xlabel("Sequence Length (T)")
    ax.set_ylabel("Latency (µs / batch)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(Ts)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(); ax.grid(True, which="both", alpha=0.25)
    ax.text(0.02, 0.97, "Shaded band = P50–P95",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")

    plt.tight_layout()
    savefig(fig, "fig4_latency_vs_seqlen")


def fig_speedup_vs_seqlen(m):
    scaling = m["latency_scaling_seq"]
    Ts      = sorted(int(k) for k in scaling)
    sp_full  = [scaling[str(T)]["speedup_full_vs_ebm"] for T in Ts]
    sp_short = [
        scaling[str(T)]["VanillaLSTM_short"]["mean"] /
        scaling[str(T)]["EBM_LSTM"]["mean"]
        for T in Ts
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("EBM-LSTM Speedup vs Sequence Length", fontsize=12)
    ax.plot(Ts, sp_full,  color=COLORS["VanillaLSTM_full"],
            lw=2, marker="o", label="vs LSTM (full window)")
    ax.plot(Ts, sp_short, color=COLORS["VanillaLSTM_short"],
            lw=2, marker="s", ls="--", label="vs LSTM (short window)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="no speedup")
    ax.set_xlabel("Sequence Length (T)")
    ax.set_ylabel("Speedup (×)")
    ax.set_xscale("log")
    ax.set_xticks(Ts)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(); ax.grid(True, which="both", alpha=0.25)

    plt.tight_layout()
    savefig(fig, "fig5_speedup_vs_seqlen")


def fig_latency_distribution(m):
    core   = m["core_comparison"]
    names  = MODEL_ORDER
    labels = [LABELS[n] for n in names]

    data = []
    for n in names:
        lat = core[n]["latency"]
        data.append(lat["all"] if "all" in lat else [lat["mean"]] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Per-Batch Inference Latency Distribution", fontsize=12)

    ax = axes[0]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS[n]); patch.set_alpha(0.7)
    ax.set_ylabel("Latency (µs / batch)"); ax.set_title("Box Plot")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    pct_keys = ["mean", "p50", "p95", "p99"]
    pct_lbls = ["Mean", "P50", "P95", "P99"]
    x     = np.arange(len(pct_keys))
    width = 0.25
    for i, n in enumerate(names):
        vals = [core[n]["latency"][k] for k in pct_keys]
        ax.bar(x + i * width, vals, width,
               color=COLORS[n], edgecolor="black", label=LABELS_SHORT[n])
    ax.set_xticks(x + width); ax.set_xticklabels(pct_lbls)
    ax.set_ylabel("Latency (µs / batch)"); ax.set_title("Latency Percentiles")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    savefig(fig, "fig6_latency_distribution")


def fig_confusion_matrices(m):
    core  = m["core_comparison"]
    names = MODEL_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Confusion Matrices (test set)", fontsize=12)

    im = None
    for ax, n in zip(axes, names):
        r  = core[n]
        cm = np.array([[r["tn"], r["fp"]],
                       [r["fn"], r["tp"]]], dtype=float)
        cm_pct = cm / cm.sum()

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title(LABELS_SHORT[n])

        for i in range(2):
            for j in range(2):
                ax.text(j, i,
                        f"{int(cm[i,j])}\n({cm_pct[i,j]:.1%})",
                        ha="center", va="center",
                        color="white" if cm_pct[i,j] > 0.5 else "black",
                        fontsize=9)

        ax.set_xlabel(
            f"acc={r['accuracy']:.3f}  F1={r['f1']:.3f}  "
            f"prec={r['precision']:.3f}  rec={r['recall']:.3f}",
            fontsize=8,
        )

    plt.colorbar(im, ax=axes[-1], fraction=0.04, pad=0.04,
                 label="Fraction of test set")
    plt.tight_layout()
    savefig(fig, "fig7_confusion_matrices")


def main():
    if not os.path.exists(METRICS):
        raise FileNotFoundError(
            f"{METRICS} not found — run collect_metrics.py first."
        )

    with open(METRICS) as f:
        m = json.load(f)

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Generating figures → {SAVE_DIR}/\n")

    fig_core_comparison(m)
    fig_tradeoff_scatter(m)
    fig_training_curves(m)
    fig_latency_vs_seqlen(m)
    fig_speedup_vs_seqlen(m)
    fig_latency_distribution(m)
    fig_confusion_matrices(m)

    print("\nDone.")


if __name__ == "__main__":
    main()
