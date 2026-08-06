#!/usr/bin/env python3
"""Regenerate every report figure and table from results/.

Visual style follows the paper's website figures (transformer-circuits Fig 24):
light gray panel, white horizontal gridlines, no axis box, bold left-aligned
title, frameless horizontal legend between title and panel, thin capped black
error bars (Wilson 95% CIs over problems), muted series colors, group brackets
under categorical axes. Numbers live in the report tables; bars are unlabelled.

Safe to re-run at any time; conditions that have not landed are skipped.
Writes reports/figures/*.{pdf,png} and reports/tables/*.tex.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

REPO = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "reports" / "figures"
TAB_DIR = REPO / "reports" / "tables"

BLUE, ORANGE, GREEN, RED = "#4c72b0", "#dd8452", "#55a868", "#c44e52"
GRAY = "#999999"
INK2 = "#555555"

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "#ebebeb",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "white",
    "grid.linestyle": "-",
    "grid.linewidth": 1.1,
    "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.bottom": False,
    "ytick.left": False,
    "text.color": "#2b2b2b",
    "axes.labelcolor": "#2b2b2b",
    "xtick.color": "#2b2b2b",
    "ytick.color": "#2b2b2b",
    "legend.frameon": False,
})

EBAR = {"lw": 1.1, "ecolor": "#1a1a1a"}


def paper_style(ax, title: str, ylabel: str | None = None, ncol: int = 2,
                legend: bool = True, pad: int = 30) -> None:
    """Bold left title; horizontal frameless legend between title and panel;
    arrowed y-label."""
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11, pad=pad)
    if ylabel:
        ax.set_ylabel(ylabel + r" $\rightarrow$", fontsize=10)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.005), ncol=ncol,
                  fontsize=9, handlelength=1.0, handleheight=1.0,
                  columnspacing=1.2, borderaxespad=0)


def pct_axis(ax, top: float = 1.05) -> None:
    ax.set_ylim(0, top)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"[{name}] written")


def load_runs() -> dict[tuple[str, str], dict]:
    """Latest run per (model, tag); smokes/probes and the truncation-
    invalidated v1 CoT runs excluded."""
    runs: dict[tuple[str, str], dict] = {}
    for summ in sorted(REPO.glob("results/*/*/summary.json")):
        run_dir = summ.parent
        if "smoke" in run_dir.name or "probe" in run_dir.name:
            continue
        try:
            summary = json.loads(summ.read_text())
            config = json.loads((run_dir / "config.json").read_text())
        except Exception:
            continue
        if "tag" not in summary:
            continue
        if (config.get("mode") == "cot"
                and config["sampling"]["max_new_tokens"] < 2048):
            continue
        key = (config.get("model", "Qwen/Qwen3-4B"), summary["tag"])
        runs[key] = {"summary": summary, "config": config, "dir": run_dir.name}
    return runs


def find(runs, model, *, contains=(), excludes=("nofilter", "random",
                                                "rankshift", "logitlens",
                                                "_sd_")):
    for (m, tag), r in runs.items():
        if m != model:
            continue
        if all(c in tag for c in contains) and not any(e in tag for e in excludes):
            return r
    return None


def acc_err(r):
    s = r["summary"]
    a = s["accuracy"]
    lo, hi = s["wilson95"]
    return a, [[a - lo], [hi - a]]


M4B = "Qwen/Qwen3-4B"


# ------------------------------------------------------------------ figure 1


def fig_gsm8k(runs) -> None:
    conds = [
        ("clean", "_none"),
        ("light", "_light"),
        ("medium", "_medium_n"),
        ("heavy", "_heavy"),
        ("random ctrl", "_medium_random"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    w = 0.38
    for k, (arm, color, label) in enumerate(
        [("cot", BLUE, "chain of thought"), ("direct", ORANGE, "direct answer")]
    ):
        for i, (name, frag) in enumerate(conds):
            excl = tuple(e for e in ("nofilter", "rankshift", "logitlens",
                                     "_sd_")) + (
                () if "random" in frag else ("random",))
            r = find(runs, M4B, contains=(f"gsm8k_{arm}{frag}",), excludes=excl)
            if r is None:
                continue
            a, err = acc_err(r)
            ax.bar(i + (k - 0.5) * (w + 0.02), a, w, color=color,
                   yerr=err, capsize=3, error_kw=EBAR,
                   label=label if i == 0 else None)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=10)
    pct_axis(ax)
    paper_style(ax, "Ablation leaves chain of thought untouched\n"
                    "and damages direct answering (GSM8K, n=200/cell)",
                "GSM8K accuracy", pad=34)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([0.65, 3.35], [-0.115, -0.115], color="#888888", lw=1,
            transform=trans, clip_on=False)
    ax.text(2.0, -0.20, "J-space ablation, three band widths",
            transform=trans, ha="center", fontsize=9, color=INK2)
    ax.text(4.0, -0.20, "norm-matched", transform=trans, ha="center",
            fontsize=9, color=INK2)
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig_gsm8k_retention")


# ------------------------------------------------------------------ figure 2


def fig_ordering() -> None:
    import random as _r

    probes = sorted(REPO.glob("results/*/*_numeric_probe/detail.jsonl"))
    if not probes:
        print("[fig_ordering] no probe detail")
        return
    pairs = []
    for line in probes[-1].read_text().splitlines():
        d = json.loads(line)
        if d.get("battery") != "A":
            continue
        fl = (d.get("first_layer_z2_allpos") or d.get("first_layer_z2")
              or d.get("first_layer_z2_final_pos"))
        for idx, tt in enumerate(d["targets"]):
            if fl[tt] is not None:
                pairs.append((idx + 1, fl[tt]))
    if len(pairs) < 8:
        print("[fig_ordering] too few pairs")
        return
    from scipy.stats import spearmanr

    rho, pval = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    rng = _r.Random(0)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter([a + rng.uniform(-0.14, 0.14) for a, _ in pairs],
               [b + rng.uniform(-0.25, 0.25) for _, b in pairs],
               s=28, color=BLUE, alpha=0.75, linewidths=0, zorder=3)
    for idx in (1, 2, 3):
        ys = sorted(b for a, b in pairs if a == idx)
        if ys:
            med = ys[len(ys) // 2]
            ax.plot([idx - 0.24, idx + 0.24], [med, med], color="#1a1a1a",
                    lw=2, zorder=4)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1st", "2nd", "3rd (answer)"])
    ax.set_xlabel("intermediate, in computation order", fontsize=10)
    paper_style(ax, "During direct answering, each intermediate result\n"
                    "appears inside the model, in computation order\n"
                    f"(rank corr. {rho:.2f}, p={pval:.0e}, n={len(pairs)})",
                "layer where the lens first sees the value",
                legend=False, pad=14)
    save(fig, "fig_numeric_ordering")


# ------------------------------------------------------------------ figure 3


def fig_scale(runs) -> None:
    models = [("Qwen/Qwen3-1.7B", "1.7B"), (M4B, "4B"), ("Qwen/Qwen3-8B", "8B")]
    rows = {"cot": [], "direct": []}
    for model, _ in models:
        for arm in ("cot", "direct"):
            c = find(runs, model, contains=(f"_{arm}_none",))
            m = find(runs, model, contains=(f"_{arm}_medium",))
            if c and m:
                base = c["summary"]["accuracy"]
                a, err = acc_err(m)
                rows[arm].append((a / base,
                                  [[err[0][0] / base], [err[1][0] / base]]))
            else:
                rows[arm].append(None)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for arm, color, label, dx in (("cot", BLUE, "chain of thought", -0.06),
                                  ("direct", ORANGE, "direct answer", 0.06)):
        xs, ys, errs = [], [], [[], []]
        for i, v in enumerate(rows[arm]):
            if v is None:
                continue
            xs.append(i + dx)
            ys.append(v[0])
            errs[0].append(v[1][0][0])
            errs[1].append(v[1][1][0])
        if xs:
            ax.errorbar(xs, ys, yerr=errs, color=color, marker="o", ms=6,
                        lw=1.8, capsize=3, label=label, zorder=3,
                        elinewidth=1.1, ecolor="#1a1a1a")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), ha="left", va="bottom",
                            fontsize=8, color=INK2, xytext=(4, 3),
                            textcoords="offset points")
    ax.axhline(1.0, ls="--", lw=1, color=GRAY)
    ax.annotate("no damage", (1.98, 1.008), fontsize=8, color=GRAY,
                ha="right", va="bottom")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[1] for m in models])
    ax.set_xlabel("model size (same fractional band, own pre-fitted lens)",
                  fontsize=10)
    paper_style(ax, "Direct-mode ablation damage shrinks with scale;\n"
                    "chain of thought is protected at every scale",
                "medium-ablated / clean", pad=34)
    save(fig, "fig_scale_retention")


# ------------------------------------------------------------------ figure 4


def fig_depth() -> None:
    probes = sorted(REPO.glob("results/*/*_depth_ladder/summary.json"))
    if not probes:
        print("[fig_depth] no depth ladder summary")
        return
    s = json.loads(probes[-1].read_text())
    Ls = sorted(int(k) for k in s["per_length"])
    acc = [s["per_length"][str(L)]["direct_acc"] for L in Ls]
    surf = [s["per_length"][str(L)]["surfacing_frac_anyband"] for L in Ls]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axvspan(2, 3, color="white", alpha=0.9, lw=0)
    ax.plot(Ls, acc, color=ORANGE, marker="s", ms=6, lw=1.8,
            label="answers correctly (no CoT)")
    ax.plot(Ls, surf, color=BLUE, marker="o", ms=6, lw=1.8,
            label="answer visible in J-lens pre-emission")
    ax.annotate("both collapse between\nL=2 and 3  (D = 2;\npre-registered 3–7)",
                (3.15, 0.58), ha="left", fontsize=8.5, color=INK2)
    ax.annotate("example, L=3:\na = 7 − 3\nb = a + 5\nc = b − 6\nwhat is c?",
                (6.1, 0.42), ha="left", fontsize=8, color=INK2,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#cccccc", lw=0.8))
    ax.axhline(1 / 9, ls=":", lw=1, color=GRAY)
    ax.annotate("digit-guess floor", (7.9, 1 / 9 + 0.015), fontsize=8,
                color=GRAY, ha="right", va="bottom")
    ax.set_xlabel("chain length L (sequential single-digit operations)",
                  fontsize=10)
    ax.set_xticks(Ls)
    pct_axis(ax, top=1.1)
    paper_style(ax, "How many operations can the model chain internally?\n"
                    "(L dependent steps, no CoT; n=30/length)",
                "fraction of chains", pad=34)
    save(fig, "fig_depth_ladder")


# ------------------------------------------------------------------ figure 5


def fig_datasets(runs) -> None:
    tasks = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("aime", "AIME")]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    w = 0.38
    plotted = False
    for k, (arm, color, label) in enumerate(
        [("cot", BLUE, "chain of thought"), ("direct", ORANGE, "direct answer")]
    ):
        first = True
        for i, (task, _) in enumerate(tasks):
            c = find(runs, M4B, contains=(f"{task}_{arm}_none",))
            m = find(runs, M4B, contains=(f"{task}_{arm}_medium",))
            x = i + (k - 0.5) * (w + 0.02)
            if c is None or m is None:
                continue
            base = c["summary"]["accuracy"]
            if base < 0.10:
                ax.annotate("undefined:\nclean = 0%", (x, 0.03), ha="center",
                            va="bottom", fontsize=8, color=INK2)
                continue
            a, err = acc_err(m)
            ax.bar(x, a / base, w, color=color,
                   yerr=[[err[0][0] / base], [err[1][0] / base]], capsize=3,
                   error_kw=EBAR, label=label if first else None)
            first = False
            plotted = True
    if not plotted:
        plt.close(fig)
        print("[fig_datasets] not enough cells yet")
        return
    ax.axhline(1.0, ls="--", lw=1, color=GRAY)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[1] for t in tasks])
    ax.set_ylim(0, 1.25)
    paper_style(ax, "Retention under medium ablation\nacross the difficulty range",
                "medium-ablated / clean", pad=34)
    save(fig, "fig_datasets_retention")


# ------------------------------------------------------------------ figure 6


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def fig_maintenance() -> None:
    probes = sorted(REPO.glob("results/*/*_rung15/summary.json"))
    if not probes:
        print("[fig_maintenance] no rung15 summary")
        return
    s = json.loads(probes[-1].read_text())
    gaps = sorted(int(g) for g in s["accuracy_by_gap"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    for mode, color, label in (("w", BLUE, "values written (CoT form)"),
                               ("u", ORANGE, "answered directly")):
        ys, lo_e, hi_e = [], [], []
        for g in gaps:
            cell = s["accuracy_by_gap"][str(g)][mode]
            a, n = cell["acc"], cell["n"]
            lo, hi = wilson(round(a * n), n)
            ys.append(a)
            lo_e.append(a - lo)
            hi_e.append(hi - a)
        ax1.errorbar(gaps, ys, yerr=[lo_e, hi_e], color=color, marker="o",
                     ms=6, lw=1.8, capsize=3, label=label,
                     elinewidth=1.1, ecolor="#1a1a1a")
    ax1.set_xlabel("reuse gap (intervening lines)", fontsize=10)
    ax1.set_xticks(gaps)
    pct_axis(ax1, top=1.1)
    paper_style(ax1, "The page is load-bearing\n(n=34/cell)", "accuracy",
                ncol=1, pad=34)

    z2 = s["z_by_gap"]["2"]
    bars = [
        ("value\n(written)", z2["z_gap_w_max"], BLUE),
        ("decoys\n(written)", z2["z_gap_w_decoy_max"], "#a8bedd"),
        ("value\n(direct)", z2["z_gap_u_max"], ORANGE),
        ("decoys\n(direct)", z2["z_gap_u_decoy_max"], "#eec4a4"),
    ]
    for i, (name, cell, color) in enumerate(bars):
        ax2.bar(i, cell["mean"], color=color, width=0.7)
        ax2.annotate(f"{cell['mean']:.2f}", (i, cell["mean"] + 0.05),
                     ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax2.set_ylim(0, 2.6)
    paper_style(ax2, "...but no idle value in the workspace\n"
                     "(gap 2; exploratory)",
                "max occupancy z in gap", legend=False, pad=34)
    fig.tight_layout()
    save(fig, "fig_maintenance")


# ------------------------------------------------------------------ figure 7


def fig_rescue() -> None:
    probes = sorted(REPO.glob("results/*/*_prefix_rescue/summary.json"))
    if not probes:
        print("[fig_rescue] no prefix_rescue summary")
        return
    s = json.loads(probes[-1].read_text())
    conds = [("own_cot", "own correct\nreasoning"),
             ("filler", "content-free filler\n(token-matched)"),
             ("none", "no prefix"),
             ("wrong_problem", "another problem's\nreasoning")]
    palette = [GREEN, BLUE, GRAY, RED]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    w = 0.19
    centers = []
    for k, abl in enumerate(["none", "medium"]):
        base = k * 1.12
        centers.append(base)
        for i, (key, name) in enumerate(conds):
            cell = s.get(f"{key}_{abl}")
            if cell is None:
                continue
            a = cell["acc"]
            lo, hi = cell["wilson95"]
            ax.bar(base + (i - 1.5) * (w + 0.015), a, w, color=palette[i],
                   yerr=[[a - lo], [hi - a]], capsize=3, error_kw=EBAR,
                   label=name.replace("\n", " ") if k == 0 else None)
    ax.set_xticks(centers)
    ax.set_xticklabels(["clean model", "medium-ablated model"], fontsize=10)
    pct_axis(ax)
    paper_style(ax, "Rescue is carried by content, not token count\n"
                    "(prefilled prefixes; n=150 problems)", "accuracy",
                pad=34)
    save(fig, "fig_prefix_rescue")


def fig_cot_correspondence() -> None:
    probes = sorted(REPO.glob("results/*/*_cot_correspondence/summary.json"))
    if not probes:
        print("[fig_cot_corr] no cot_correspondence summary")
        return
    s = json.loads(probes[-1].read_text())
    windows = [("z_write", "while its line\nis written"),
               ("z_post", "while later lines\nare written"),
               ("z_use", "where it\nis used")]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, (k, label) in enumerate(windows):
        cell = s[k]
        e = 1.96 * cell["se"]
        ax.bar(i, cell["mean"], 0.62, color=BLUE,
               yerr=[[e], [e]], capsize=3, error_kw=EBAR)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([w[1] for w in windows], fontsize=9)
    ax.set_xlabel("position in the model's own transcript", fontsize=9)
    ax.axhline(0.0, color="#888888", lw=0.8)
    paper_style(ax, "In its own chain of thought, a value\n"
                    "occupies the workspace only in use",
                "occupancy z of the value", legend=False, pad=34)
    fig.tight_layout()
    save(fig, "fig_cot_correspondence")


# ---------------------------------------------------------- verdicts panel


def fig_verdicts() -> None:
    """Four mechanism hypotheses as cards with colour-coded verdicts."""
    cards = [
        ("EXCHANGE", "writing frees\nthe workspace",
         "test. does a value's internal\npresence dip once written,\n"
         "until it is reused?",
         "OPEN", "design confound found.\nsource text stays readable\n"
         "in context in both modes", "#888888"),
        ("DUPLICATION", "the page is a\nredundant backup",
         "test. is a written value\nstill held internally\nuntil reused?",
         "OPEN", "same confound.\nno verdict possible from\n"
         "single-context designs", "#888888"),
        ("COMPLEMENTARY", "page and workspace\nhold different content",
         "test. are CoT values absent\nfrom the direct-mode\nreadout?",
         "REJECTED", "values are present,\nin computation order\n"
         "(rank corr. 0.70, p=2e-8)", RED),
        ("EPIPHENOMENAL", "CoT helps via extra\ncompute, not content",
         "test. does matched-length\nfiller rescue an ablated\nmodel?",
         "REJECTED", "no. filler 23% = baseline\n19%, real reasoning 98%\n"
         "(n=150)", RED),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
    for ax, (name, claim, test, verdict, why, vcol) in zip(axes, cards):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True,
                                   facecolor="#f4f4f2", edgecolor="#cccccc",
                                   lw=1.2, zorder=1))
        ax.text(0.5, 0.90, name, ha="center", va="top", fontsize=11,
                fontweight="bold", zorder=3)
        ax.text(0.5, 0.79, claim, ha="center", va="top", fontsize=9,
                style="italic", color="#444444", zorder=3)
        ax.text(0.5, 0.585, test, ha="center", va="top", fontsize=8,
                color="#555555", zorder=3)
        ax.add_patch(plt.Rectangle((0.20, 0.255), 0.60, 0.105, fill=True,
                                   facecolor=vcol, edgecolor="none", zorder=2))
        ax.text(0.5, 0.307, verdict, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=3)
        ax.text(0.5, 0.215, why, ha="center", va="top", fontsize=7.5,
                color="#555555", zorder=3)
    fig.suptitle("What does externalisation mean? Four mechanisms, "
                 "tested with pre-registered evidence",
                 fontsize=12, fontweight="bold", x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig_verdicts")


# ------------------------------------------------------------------- table


def table_results(runs) -> None:
    rows = []
    for (model, tag) in sorted(runs):
        s = runs[(model, tag)]["summary"]
        lo, hi = s["wilson95"]
        slug = model.split("/")[-1]
        tex_tag = f"{slug}: {tag}".replace("_", "\\_")
        rows.append(
            f"{tex_tag} & {s['n']} & {s['accuracy']:.3f} & "
            f"[{lo:.3f}, {hi:.3f}] & {s['mean_new_tokens']:.0f} & "
            f"{s['stop_reasons'].get('length', 0)} \\\\"
        )
    body = "\n".join(rows)
    (TAB_DIR / "all_runs.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "condition & $n$ & acc & 95\\% CI & mean tokens & len-stops \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )
    print(f"[table] {len(rows)} rows written")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    print(f"[figures] {len(runs)} runs loaded")
    fig_gsm8k(runs)
    fig_ordering()
    fig_scale(runs)
    fig_depth()
    fig_datasets(runs)
    fig_maintenance()
    fig_rescue()
    fig_cot_correspondence()
    fig_verdicts()
    table_results(runs)


if __name__ == "__main__":
    main()
