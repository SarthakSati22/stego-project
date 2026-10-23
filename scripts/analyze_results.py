"""Tables and plots from results/experiments_<profile>.csv."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGIES = ["audio_only", "video_only", "fixed_50_50",
              "adaptive_no_feedback", "adaptive_with_feedback"]
CHANNELS = ["none", "mp3_128", "mp3_64", "h264_crf23", "h264_crf28", "h265_crf23",
            "h265_crf28", "drop_5", "drop_10", "resize_75",
            "mp3_128+h264_crf23", "mp3_64+h264_crf28"]
KEYS = ["clip_id", "payload", "seed"]


def paired_ci(runs, a, b, col, n_boot=2000, seed=0):
    """Mean of (a - b) over matched runs, with a bootstrap 95% interval."""
    x = runs[runs["strategy"] == a].set_index(KEYS)[col]
    y = runs[runs["strategy"] == b].set_index(KEYS)[col]
    d = (x - y.reindex(x.index)).dropna().to_numpy()
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.random.default_rng(seed).choice(d, size=(n_boot, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "robust"
    df = pd.read_csv(f"results/experiments_{profile}.csv")
    for c in ("snr", "psnr"):
        df.loc[~np.isfinite(df[c]), c] = np.nan
    fig_dir, tab_dir = Path("results/figures"), Path("results/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    runs = df[df["channel"] == "none"].copy()          # one row per (clip, payload, seed, strategy)
    rec_att = (df[df["channel"] != "none"]
               .groupby(KEYS + ["strategy"])["recovered"].mean().rename("rec_attacked"))
    runs = runs.merge(rec_att.reset_index(), on=KEYS + ["strategy"])
    runs["alloc_flag"] = runs["alloc_p_fused"] > 0.5
    runs["judge_flag"] = runs["judge_p_fused"] > 0.5

    channels = [c for c in CHANNELS if c in set(df["channel"])]
    rec = (df.pivot_table(index="strategy", columns="channel", values="recovered", aggfunc="mean")
           .reindex(index=STRATEGIES, columns=channels))
    summary = runs.groupby("strategy").agg(
        audio_share=("share", "mean"), snr_db=("snr", "mean"), psnr_db=("psnr", "mean"),
        ssim=("ssim", "mean"), flagged_allocator_det=("alloc_flag", "mean"),
        flagged_judge=("judge_flag", "mean"), judge_score=("judge_p_fused", "mean"),
        recovery_attacked=("rec_attacked", "mean")).reindex(STRATEGIES)

    rows = []
    for other in STRATEGIES[:-1]:
        for col, label in (("judge_p_fused", "detect score (judge)"),
                           ("alloc_p_fused", "detect score (allocator detectors)"),
                           ("rec_attacked", "recovery under attack")):
            m, lo, hi = paired_ci(runs, "adaptive_with_feedback", other, col)
            rows.append(dict(vs=other, metric=label, mean_diff=m, ci_low=lo, ci_high=hi))
    paired = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    text = ("RECOVERY RATE (strategy x channel)\n" + rec.round(3).to_string()
            + "\n\nSUMMARY PER STRATEGY\n" + summary.round(3).to_string()
            + "\n\nADAPTIVE WITH FEEDBACK MINUS OTHER STRATEGY (paired, 95% bootstrap interval)\n"
            "  detect score: negative = adaptive is LESS detectable (good)\n"
            "  recovery:     positive = adaptive recovers MORE (good)\n"
            + paired.round(4).to_string(index=False))
    print(text)
    (tab_dir / f"summary_{profile}.txt").write_text(text, encoding="utf-8")
    rec.to_csv(tab_dir / f"recovery_{profile}.csv")
    summary.to_csv(tab_dir / f"summary_{profile}.csv")
    paired.to_csv(tab_dir / f"paired_{profile}.csv", index=False)

    # Figure 1: recovery heatmap
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(rec.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(rec.shape[1]), rec.columns, rotation=40, ha="right")
    ax.set_yticks(range(rec.shape[0]), rec.index)
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            v = rec.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="recovery rate")
    ax.set_title(f"Secret recovery by strategy and channel ({profile} profile)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"recovery_heatmap_{profile}.png", dpi=150)
    plt.close(fig)

    # Figure 2: how often each strategy is flagged
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(STRATEGIES))
    ax.bar(x - 0.2, summary["flagged_allocator_det"], 0.4, label="allocator's detectors")
    ax.bar(x + 0.2, summary["flagged_judge"], 0.4, label="independent judge")
    ax.set_xticks(x, STRATEGIES, rotation=20, ha="right")
    ax.set_ylabel("share of stego clips flagged")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Detectability by strategy (lower is better)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"detectability_{profile}.png", dpi=150)
    plt.close(fig)

    # Figure 3: robustness versus detectability
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in STRATEGIES:
        ax.scatter(summary.loc[s, "judge_score"], summary.loc[s, "recovery_attacked"], s=60)
        ax.annotate(s, (summary.loc[s, "judge_score"], summary.loc[s, "recovery_attacked"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean detection score (judge, lower is better)")
    ax.set_ylabel("recovery under attack (higher is better)")
    ax.set_title("Trade-off: stealth vs robustness")
    fig.tight_layout()
    fig.savefig(fig_dir / f"tradeoff_{profile}.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved tables to {tab_dir} and figures to {fig_dir}")


if __name__ == "__main__":
    main()