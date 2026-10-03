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

BLUE, ORANGE, GREEN, RED = "#4c72b0", "#c05f10", "#55a868", "#c44e52"
GRAY = "#999999"
INK2 = "#555555"


def tint(hexcolor: str, f: float = 0.62) -> str:
    """Lighter step of the same hue, for clean-vs-ablated pairs."""
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * f) for c in (r, g, b))


LIGHT_BLUE, LIGHT_ORANGE = tint(BLUE), tint(ORANGE)
DOT = {"color": "#1a1a1a", "edgecolors": "white", "linewidths": 0.8,
       "zorder": 3}

plt.rcParams.update({
    "font.size": 12,
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

EBAR = {"lw": 1.3, "ecolor": "#1a1a1a"}


def bar_label(ax, x, v, fontsize=9):
    ax.annotate(f"{v * 100:.0f}", (x, v + 0.02), ha="center", va="bottom",
                fontsize=fontsize, color=INK2)


def sig_bracket(ax, x1, x2, y, text, tick=0.025, fontsize=8.5):
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#1a1a1a",
            lw=1.1, clip_on=False)
    ax.annotate(text, ((x1 + x2) / 2, y + 0.01), ha="center", va="bottom",
                fontsize=fontsize, color="#1a1a1a")


def paper_style(ax, title: str, ylabel: str | None = None, ncol: int = 2,
                legend: bool = True, pad: int = 30) -> None:
    """Bold left title; horizontal frameless legend between title and panel;
    arrowed y-label."""
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=pad)
    if ylabel:
        ax.set_ylabel(ylabel + r" $\rightarrow$", fontsize=11)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.005), ncol=ncol,
                  fontsize=9.5, handlelength=1.0, handleheight=1.0,
                  columnspacing=1.2, borderaxespad=0)


def pct_axis(ax, top: float = 1.05) -> None:
    ax.set_ylim(0, top)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300)
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
        tag = summary["tag"]
        argv = " ".join(map(str, config.get("argv", [])))
        if "--k 5" in argv:
            tag += "_k5"
        elif "--k 20" in argv:
            tag += "_k20"
        key = (config.get("model", "Qwen/Qwen3-4B"), tag)
        runs[key] = {"summary": summary, "config": config, "dir": run_dir.name}
    return runs


def find(runs, model, *, contains=(), excludes=("nofilter", "random",
                                                "rankshift", "logitlens",
                                                "_sd_", "_s1", "_s2",
                                                "_s3", "_s4")):
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
        ("random\nctrl", "_medium_random"),
        ("rank-\nshift ctrl", "_medium_jlens_rankshift"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    w = 0.72
    ticks, labels = [], []
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for k, (arm, hue, cluster) in enumerate(
        [("cot", BLUE, "chain of thought"),
         ("direct", ORANGE, "direct answer")]
    ):
        base = k * (len(conds) + 1.2)
        for i, (name, frag) in enumerate(conds):
            excl = ("nofilter", "logitlens", "_sd_", "_s1", "_s2", "_s3",
                    "_s4") + (
                () if "rankshift" in frag else ("rankshift",)) + (
                () if "random" in frag else ("random",))
            r = find(runs, M4B, contains=(f"gsm8k_{arm}{frag}",), excludes=excl)
            if r is None:
                continue
            a, err = acc_err(r)
            color = (tint(hue) if name == "clean"
                     else GRAY if "ctrl" in name else hue)
            ax.bar(base + i, a, w, color=color, yerr=err, capsize=3,
                   error_kw=EBAR)
            ticks.append(base + i)
            labels.append(name)
        ax.text(base + 2.5, -0.30, cluster, transform=trans, ha="center",
                fontsize=11, fontweight="bold", color="#2b2b2b")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    pct_axis(ax)
    paper_style(ax, "Ablation leaves chain of thought untouched\n"
                    "and damages direct answering (GSM8K, n=200/cell)",
                "GSM8K accuracy", legend=False, pad=14)
    fig.subplots_adjust(bottom=0.28)
    save(fig, "fig_gsm8k_retention")


# ------------------------------------------------------------------ figure 2


def fig_ordering() -> None:
    import random as _r

    probes = sorted(REPO.glob("results/*/*_numeric_probe/detail.jsonl"))
    if not probes:
        print("[fig_ordering] no probe detail")
        return
    import itertools as _it
    pairs = []
    _lines = _it.chain.from_iterable(
        _pf.read_text().splitlines() for _pf in probes[-3:])
    for line in _lines:
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
               s=30, color=BLUE, alpha=0.75, edgecolors="white",
               linewidths=0.6, zorder=3)
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
    models = [("Qwen/Qwen3-1.7B", "1.7B"), (M4B, "4B"), ("Qwen/Qwen3-8B", "8B"),
              ("Qwen/Qwen3-14B", "14B")]
    rows = {"cot": [], "direct": []}
    for model, _ in models:
        for arm in ("cot", "direct"):
            c = find(runs, model, contains=("gsm8k", f"_{arm}_none"))
            m = find(runs, model, contains=("gsm8k", f"_{arm}_medium"))
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
            off = (4, -13) if arm == "cot" else (4, 6)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), ha="left",
                            va="bottom" if arm != "cot" else "top",
                            fontsize=8, color=INK2, xytext=off,
                            textcoords="offset points")
    ax.axhline(1.0, ls="--", lw=1, color=GRAY)
    ax.annotate("no damage", (-0.28, 1.012), fontsize=8, color=GRAY,
                ha="left", va="bottom")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[1] for m in models])
    ax.set_xlabel("model size (same fractional band, own pre-fitted lens)",
                  fontsize=10)
    paper_style(ax, "Chain of thought is protected at every scale\n"
                    "(GSM8K, n=200/cell, each model with its own "
                    "pre-fitted lens)",
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
    ax.annotate("both collapse between\nL=2 and 3  (D = 2,\npre-registered 3–7)",
                (3.15, 0.58), ha="left", fontsize=8.5, color=INK2)
    ax.annotate("example at L=3\na = 7 − 3\nb = a + 5\nc = b − 6\nwhat is c?",
                (6.1, 0.42), ha="left", fontsize=8, color=INK2,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#cccccc", lw=0.8))
    ax.axhline(1 / 9, ls=":", lw=1, color=GRAY)
    ax.annotate("digit-guess floor", (6.85, 1 / 9 + 0.015), fontsize=8,
                color=GRAY, ha="right", va="bottom")
    ax.set_xlabel("chain length L (sequential single-digit operations)",
                  fontsize=10)
    ax.set_xticks(Ls)
    pct_axis(ax, top=1.1)
    paper_style(ax, "How many operations can the model chain internally?\n"
                    "(L dependent steps, no CoT, n=30/length)",
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
                ax.annotate("undefined\n(clean = 0%)", (x, 0.03), ha="center",
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


def fig_datasets_accuracy(runs) -> None:
    """Table 1 as bars: raw accuracies, clean vs medium-ablated, both arms,
    across the difficulty range. Same encoding as fig_task_balance."""
    tasks = [("gsm8k", "GSM8K\n(n=200)"), ("math500", "MATH-500\n(n=243)"),
             ("aime", "AIME\n(n=60)")]
    cells = (("cot", "none", LIGHT_BLUE, "CoT, clean"),
             ("cot", "medium", BLUE, "CoT, ablated"),
             ("direct", "none", LIGHT_ORANGE, "direct, clean"),
             ("direct", "medium", ORANGE, "direct, ablated"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    w = 0.2
    for k, (arm, abl, color, label) in enumerate(cells):
        for i, (task, _) in enumerate(tasks):
            frag = "_none" if abl == "none" else "_medium"
            r = find(runs, M4B, contains=(f"{task}_{arm}{frag}",))
            if r is None:
                continue
            a, err = acc_err(r)
            ax.bar(i + (k - 1.5) * (w + 0.005), a, w, color=color,
                   yerr=err, capsize=2.5, error_kw=EBAR,
                   label=label if i == 0 else None)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[1] for t in tasks], fontsize=10)
    pct_axis(ax)
    paper_style(ax, "Accuracy across the difficulty range\n"
                    "(Qwen3-4B, clean vs medium-ablated)",
                "accuracy", pad=34)
    fig.tight_layout()
    save(fig, "fig_datasets_accuracy")


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
        ("decoys\n(written)", z2["z_gap_w_decoy_max"], LIGHT_BLUE),
        ("value\n(direct)", z2["z_gap_u_max"], ORANGE),
        ("decoys\n(direct)", z2["z_gap_u_decoy_max"], LIGHT_ORANGE),
    ]
    for i, (name, cell, color) in enumerate(bars):
        ax2.bar(i, cell["mean"], color=color, width=0.7)
        ax2.annotate(f"{cell['mean']:.2f}", (i, cell["mean"] + 0.05),
                     ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax2.set_ylim(0, 2.6)
    paper_style(ax2, "...but no idle value in the workspace\n"
                     "(gap 2, exploratory)",
                "max occupancy z in gap", legend=False, pad=34)
    fig.tight_layout()
    save(fig, "fig_maintenance")


# ------------------------------------------------------------------ figure 7


def fig_rescue() -> None:
    probes = sorted(REPO.glob("results/*/*_prefix_rescue/summary.json"))
    probes = [p for p in probes
              if "filler_dots_medium" not in json.loads(p.read_text())]
    if not probes:
        print("[fig_rescue] no matched-subset rescue summary")
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
                    "(prefilled prefixes, n=150 problems)", "accuracy",
                pad=34)
    save(fig, "fig_prefix_rescue")


def fig_cot_correspondence() -> None:
    cands = sorted(REPO.glob("results/*/*_cot_correspondence/summary.json"))
    summaries = []
    for c in cands:
        cfg = json.loads((c.parent / "config.json").read_text())
        if cfg.get("final_variant", "var") != "var":
            continue
        if "4B" in str(cfg.get("model", "")) and "14B" not in str(cfg.get("model", "")):
            summaries.append(json.loads(c.read_text()))
    if not summaries:
        print("[fig_cot_corr] no cot_correspondence summary")
        return
    s = summaries[-1]
    windows = [("z_write", "while its line\nis written"),
               ("z_post", "while later lines\nare written"),
               ("z_use", "where it\nis used")]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for i, (k, label) in enumerate(windows):
        vals = [ss[k]["mean"] for ss in summaries]
        m = sum(vals) / len(vals)
        ax.bar(i, m, 0.62, color=BLUE)
        ax.scatter([i] * len(vals), vals, s=26, **DOT)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([w[1] for w in windows], fontsize=9)
    ax.set_xlabel("position in the model's own transcript", fontsize=9)
    ax.axhline(0.0, color="#888888", lw=0.8)
    paper_style(ax, "In its own chain of thought, a value\n"
                    "occupies the workspace only in use",
                "occupancy z of the value", legend=False, pad=34)
    fig.tight_layout()
    save(fig, "fig_cot_correspondence")


def fig_correspondence_contrast() -> None:
    """CoT vs direct on budget-matched programs (d = a op literal): the same
    three windows, both arms. Tests whether externalisation causes the fade."""
    arms = {"cot": [], "direct": []}
    for pat, arm in (("*_cot_correspondence", "cot"),
                     ("*_direct_correspondence", "direct")):
        for c in sorted(REPO.glob(f"results/*/{pat}/summary.json")):
            cfg = json.loads((c.parent / "config.json").read_text())
            if cfg.get("final_variant") != "lit":
                continue
            if arm == "direct" and "prefill" not in str(cfg.get("decoding")):
                continue  # v1-v3 direct runs failed the accuracy check
            if "4B" in str(cfg.get("model", "")):
                arms[arm].append(json.loads(c.read_text()))
    if not (arms["cot"] and arms["direct"]):
        print("[fig_corr_contrast] lit-variant runs missing")
        return
    windows = [("z_write", "at its line"),
               ("z_post", "during the two\nunrelated lines"),
               ("z_use", "where it\nis used")]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    w = 0.38
    for k, (arm, color, label) in enumerate(
            (("cot", BLUE, "chain of thought (line emitted)"),
             ("direct", ORANGE, "direct (line only read)"))):
        for i, (key, _) in enumerate(windows):
            vals = [s[key]["mean"] for s in arms[arm]]
            m = sum(vals) / len(vals)
            x = i + (k - 0.5) * (w + 0.02)
            ax.bar(x, m, w, color=color, label=label if i == 0 else None)
            ax.scatter([x] * len(vals), vals, s=24, **DOT)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([wd[1] for wd in windows], fontsize=9)
    ax.axhline(0.0, color="#888888", lw=0.8)
    ax.set_ylim(-0.35, 1.45)
    paper_style(ax, "The fade is not caused by writing. Without a page\n"
                    "the workspace still stores nothing between uses\n"
                    "(budget-matched programs, 3 seeds per arm)",
                "occupancy z of the value", ncol=1, pad=48)
    fig.tight_layout()
    save(fig, "fig_correspondence_contrast")


# ------------------------------------------------- extension-campaign figures

def _cell_acc(runs, model, contains):
    r = find(runs, model, contains=contains)
    if not r:
        return None
    return r["summary"].get("accuracy", r["summary"].get("acc"))


def fig_seed_rho() -> None:
    """Per-seed protection ratios for GSM8K and MATH-500 (five seeds each)."""
    import collections
    data = {"GSM8K": {}, "MATH-500": {}}
    for d in sorted(REPO.glob("results/*/*_s[0-4]")):
        b = d.name
        host = d.parent.name
        if any(x in b for x in ("m1.7b", "m8b", "m14b", "llama", "nofilter",
                                "random", "rankshift", "sd_", "aime")):
            continue
        cfgp = d / "config.json"
        if not cfgp.exists() or not (d / "summary.json").exists():
            continue
        cfg = json.loads(cfgp.read_text())
        argv = " ".join(map(str, cfg.get("argv", [])))
        if "--k 5" in argv or "--k 20" in argv or "--direction-source" in argv                 or "--no-filter" in argv:
            continue
        task = "GSM8K" if "gsm8k" in b else ("MATH-500" if "math500" in b else None)
        if task is None:
            continue
        seed = b.rsplit("_s", 1)[-1]
        arm = ("cot" if "_cot_" in b else "direct") + "/" +               ("none" if "_none_" in b else "medium")
        s = json.loads((d / "summary.json").read_text())
        data[task].setdefault(seed, {})[arm] = s.get("accuracy", s.get("acc"))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, (task, color) in enumerate((("GSM8K", BLUE), ("MATH-500", GREEN))):
        rhos = []
        for seed, cells in sorted(data[task].items()):
            if len(cells) == 4 and cells["direct/none"]:
                rho = (cells["cot/medium"] / cells["cot/none"]) /                       (cells["direct/medium"] / cells["direct/none"])
                rhos.append(rho)
        xs = [i + (j - len(rhos) / 2) * 0.07 for j in range(len(rhos))]
        ax.scatter(xs, rhos, color=color, s=46, zorder=3, edgecolors="white",
                   linewidths=0.8, label=f"{task} (n={len(rhos)} seeds)")
        if rhos:
            m = sum(rhos) / len(rhos)
            ax.hlines(m, i - 0.3, i + 0.3, color=color, lw=2)
            ax.annotate(f"mean {m:.2f}", (i + 0.32, m), va="center", fontsize=9)
    ax.axhline(1.0, color="#888888", lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GSM8K", "MATH-500"], fontsize=10)
    ax.set_ylim(bottom=0.8)
    paper_style(ax, "Protection replicates across five independent seeds",
                "protection ratio \u03c1 per seed", pad=30)
    fig.tight_layout()
    save(fig, "fig_seed_rho")


def fig_model_probes() -> None:
    """Masked-maintenance across model families: masked vs severed vs page.
    Wilson CIs per cell; per-model exact sign test, masked vs severed."""
    from math import comb

    entries = []
    for d in sorted(REPO.glob("results/*/*_masked_maintenance")):
        s = json.loads((d / "summary.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        m = str(cfg.get("model", ""))
        label = ("Qwen3-14B" if "14B" in m else "Qwen3-4B" if "4B" in m
                 else "Llama-3.1-8B" if "lama" in m or "Llama" in m else m)
        entries.append((label, s, d))
    seen = {}
    for label, s, d in entries:
        seen[label] = (s, d)  # latest wins
    order = [l for l in ("Qwen3-4B", "Qwen3-14B", "Llama-3.1-8B") if l in seen]

    def mcnemar_masked_severed(run_dir):
        b01 = b10 = 0
        for line in (run_dir / "detail.jsonl").read_text().splitlines():
            r = json.loads(line)
            a, b = r["masked_use"]["correct"], r["severed"]["correct"]
            b01 += a and not b
            b10 += b and not a
        n = b01 + b10
        if n == 0:
            return 1.0
        k = min(b01, b10)
        return min(1.0, sum(comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    w = 0.2
    for k, (cond, color, label) in enumerate(
            (("severed", RED, "severed"),
             ("baseline", GRAY, "baseline"),
             ("masked_use", BLUE, "definition masked"),
             ("page", GREEN, "written copy visible"))):
        for i, l in enumerate(order):
            cell = seen[l][0][cond]
            v = cell["acc"]
            lo, hi = cell["wilson95"]
            x = i + (k - 1.5) * (w + 0.01)
            ax.bar(x, v, w, color=color, yerr=[[v - lo], [hi - v]],
                   capsize=2.5, error_kw=EBAR,
                   label=label if i == 0 else None)
    for i, l in enumerate(order):
        p = mcnemar_masked_severed(seen[l][1])
        sig_bracket(ax, i - 1.5 * (w + 0.01), i + 0.5 * (w + 0.01), 1.10,
                    f"masked vs severed\n$p={p:.2f}$", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    pct_axis(ax, top=1.30)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    paper_style(ax, "The model doesn't store representations across "
                    "token positions\n(n=96 per cell)",
                "accuracy", pad=34)
    fig.tight_layout()
    save(fig, "fig_model_probes")


def fig_filler_battery() -> None:
    """Full-set prefix battery, ablated arm."""
    cand = sorted(REPO.glob("results/*/2026090*_prefix_rescue"))
    src = None
    for d in reversed(cand):
        if (d / "summary.json").exists() and                 (d / "filler_dots_medium.jsonl").exists():
            src = d
            break
    if src is None:
        print("[fig_battery] no battery run")
        return
    s = json.loads((src / "summary.json").read_text())
    conds = [("wrong_problem", "a different\nproblem's correct\nreasoning"),
             ("none", "empty\nscratchpad"),
             ("filler_dots", "\". \"\nrepeated"),
             ("filler_shuffled", "own reasoning,\ntokens\nshuffled"),
             ("filler", "\"let me think\nstep by step\"\nlooped"),
             ("own_cot", "own correct\nreasoning,\nreplayed")]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    batteries = []
    for d in cand:
        try:
            ss = json.loads((d / "summary.json").read_text())
            if "filler_dots_medium" in ss:
                batteries.append(ss)
        except Exception:
            pass
    for i, (key, label) in enumerate(conds):
        cells = [b[f"{key}_medium"] for b in batteries
                 if f"{key}_medium" in b]
        if not cells:
            continue
        a = (sum(round(c["acc"] * c["n"]) for c in cells)
             / sum(c["n"] for c in cells))
        # CI on the number of distinct problems, not the pooled (item, battery)
        # count: the three batteries reuse the same problem set
        N = cells[0]["n"]
        lo, hi = wilson(round(a * N), N)
        ax.bar(i, a, 0.9, color=GREEN if key == "own_cot" else
               (RED if key == "wrong_problem" else BLUE),
               yerr=[[a - lo], [hi - a]], capsize=3, error_kw=EBAR)
    sig_bracket(ax, 1, 3, 0.35, "best filler vs empty  $p=0.47$", fontsize=8)
    sig_bracket(ax, 4, 5, 1.07, "$p=2.4{\\times}10^{-125}$", fontsize=7.5)
    sig_bracket(ax, 3, 5, 1.17, "$p=3.8{\\times}10^{-124}$", fontsize=7.5)
    sig_bracket(ax, 2, 5, 1.27, "$p=1.4{\\times}10^{-129}$", fontsize=7.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[1] for c in conds], fontsize=8.5)
    pct_axis(ax, top=1.38)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    paper_style(ax, "No filler type rescues the ablated model\n"
                    "(Qwen3-4B, 170 GSM8K problems, medium ablation, "
                    "fillers length-matched)",
                "accuracy", pad=34)
    fig.tight_layout()
    save(fig, "fig_filler_battery")


def fig_task_balance() -> None:
    cand = sorted(REPO.glob("results/*/2026090*_task_balance"))
    src = None
    for d in reversed(cand):
        s = json.loads((d / "summary.json").read_text())
        if s.get("arith_cot_none", {}).get("n", 0) >= 80:
            src = d
            break
    if src is None:
        print("[fig_taskbal] no full task balance run")
        return
    s = json.loads((src / "summary.json").read_text())
    suites = ["arith", "recall", "transitive", "verbal"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.19
    cells = (("cot", "none", LIGHT_BLUE, "CoT, clean"),
             ("cot", "medium", BLUE, "CoT, ablated"),
             ("direct", "none", LIGHT_ORANGE, "direct, clean"),
             ("direct", "medium", ORANGE, "direct, ablated"))
    for k, (arm, abl, color, label) in enumerate(cells):
        for i, suite in enumerate(suites):
            a = s[f"{suite}_{arm}_{abl}"]["acc"]
            ax.bar(i + (k - 1.5) * (w + 0.01), a, w, color=color,
                   label=label if i == 0 else None)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels(["arithmetic\n(direct near floor)", "recall",
                        "ordering", "verbal"], fontsize=9)
    pct_axis(ax)
    paper_style(ax, "Ablation costs nothing on four synthetic task types\n"
                    "(accuracies, n=80 per suite)", "accuracy", pad=34)
    fig.tight_layout()
    save(fig, "fig_task_balance")




def fig_masking_schematic() -> None:
    """Slide schematic for the masked-maintenance design. Row colours match
    the condition colours in fig_model_probes."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    rows = [
        ("baseline", GRAY, [], False, "sanity check, everything readable"),
        ("definition\nmasked", BLUE, ["q"], False,
         "the value must travel\nthrough the gap, or fail"),
        ("severed", RED, ["q", "gap"], False, "no path at all, the floor"),
        ("written copy\n(page)", GREEN, ["q"], True,
         "a visible copy inside\nthe gap should rescue"),
    ]
    boxes = {"def": (0.16, 0.315), "gap": (0.36, 0.60), "q": (0.65, 0.92)}
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 1.28)
    ax.set_ylim(0, 1)
    ax.axis("off")
    row_h, box_h = 0.25, 0.115
    for r, (name, color, blocked, restate, note) in enumerate(rows):
        y0 = 0.89 - r * row_h - box_h
        yc = y0 + box_h / 2
        ax.text(0.135, yc, name, ha="right", va="center", fontsize=10,
                fontweight="bold", color=color)
        texts = {"def": "x = 7.",
                 "gap": "filler text ...\n+ remember x = 7." if restate
                        else "filler text\n(no digits)",
                 "q": "what is x?\nanswer, x ="}
        for key, (xa, xb) in boxes.items():
            fc = "#e7f0e7" if (key == "gap" and restate) else "white"
            ax.add_patch(FancyBboxPatch(
                (xa, y0), xb - xa, box_h, boxstyle="round,pad=0.008",
                facecolor=fc, edgecolor="#aaaaaa", lw=1.1))
            ax.text((xa + xb) / 2, yc, texts[key], ha="center", va="center",
                    fontsize=8, family="monospace", color="#2b2b2b")
        for src, rad, tag in (("gap", 0.5, "gap"), ("q", 0.28, "q")):
            bad = tag in blocked
            x_from = (boxes[src][0] + boxes[src][1]) / 2
            x_to = (boxes["def"][0] + boxes["def"][1]) / 2 + \
                   (0.045 if src == "q" else -0.025)
            arr = FancyArrowPatch(
                (x_from, y0 + box_h), (x_to, y0 + box_h),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>", mutation_scale=10,
                color=RED if bad else "#777777",
                ls=(0, (3, 2)) if bad else "-", lw=1.3)
            ax.add_patch(arr)
            if bad:
                xm = (x_from + x_to) / 2
                ym = y0 + box_h + rad * (x_from - x_to) / 2 + 0.012
                ax.text(xm, ym, "X", ha="center", va="center", fontsize=11,
                        fontweight="bold", color=RED)
        ax.text(0.95, yc, note, ha="left", va="center", fontsize=8,
                color=INK2)
    ax.text(0.135, 0.955, "attention mask, who may read the definition",
            ha="right", va="bottom", fontsize=8, color=INK2, style="italic")
    ax.set_title("The masked-maintenance design", loc="left",
                 fontweight="bold", fontsize=12, pad=16)
    save(fig, "fig_masking_schematic")


def fig_knockout() -> None:
    cand = sorted(REPO.glob("results/*/*_knockout_curve/summary.json"))
    if not cand:
        print("[fig_knockout] none")
        return
    s = json.loads(cand[-1].read_text())
    lvls = [("L0", "nothing\nmasked"), ("L1", "written values\n(+-1)"),
            ("L2", "+-3"), ("L3", "+-6"), ("L4", "entire\ntranscript")]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    xs = list(range(len(lvls)))
    ys = [s[l]["acc"] for l, _ in lvls]
    errs = [[s[l]["acc"] - s[l]["wilson95"][0] for l, _ in lvls],
            [s[l]["wilson95"][1] - s[l]["acc"] for l, _ in lvls]]
    ax.errorbar(xs, ys, yerr=errs, color=BLUE, marker="o", ms=7, lw=2,
                capsize=3, elinewidth=1.1, ecolor="#1a1a1a",
                label="value neighbourhood masked")
    rys = [s["R" + str(i)]["acc"] for i in (1, 2, 3)]
    ax.plot([1, 2, 3], rys, color=GRAY, marker="s", ms=6, lw=1.6,
            label="random positions, matched size")
    ax.axhline(s["L4"]["acc"], color=RED, lw=0.9, ls="--")
    ax.annotate("direct re-derivation floor", (0.02, s["L4"]["acc"] + 0.02),
                fontsize=8.5, color=RED)
    ax.set_xticks(xs)
    ax.set_xticklabels([l[1] for l in lvls], fontsize=9)
    pct_axis(ax)
    paper_style(ax, "The value lives in the written context, nowhere else\n"
                    "(knockout curve, n=70 transcripts)", "accuracy", pad=34)
    fig.tight_layout()
    save(fig, "fig_knockout")




def fig_injection() -> None:
    cand = sorted(REPO.glob("results/*/*_injection_sufficiency/summary.json"))
    if not cand:
        print("[fig_injection] none")
        return
    s = json.loads(cand[-1].read_text())
    bars = [("masked", "no injection\n(floor)", GRAY),
            ("inj_correct_0.1", "inject value\nweak", BLUE),
            ("inj_correct_0.3", "inject value\nright dose", BLUE),
            ("inj_correct_1.0", "inject value\noverdose", BLUE)]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, (k, label, color) in enumerate(bars):
        ax.bar(i, s[k]["acc"], 0.62, color=color)
        bar_label(ax, i, s[k]["acc"])
    tr = s["inj_wrong_0.3"]["tracks_injected"]
    ax.annotate(f"with a wrong digit planted, the model\nreports it {tr*100:.0f}% of the time",
                (3.45, 0.62), fontsize=9, ha="right", color=INK2)
    ax.axhline(s["severed"]["acc"], color=RED, lw=1.1, ls="--")
    ax.annotate("severed floor", (3.45, s["severed"]["acc"] + 0.025),
                fontsize=8.5, color=RED, ha="right", va="bottom")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[1] for b in bars], fontsize=9)
    pct_axis(ax)
    paper_style(ax, "A planted internal copy works. The model just\n"
                    "never plants one itself (n=96 per cell)", "accuracy",
                pad=34)
    fig.tight_layout()
    save(fig, "fig_injection")


# ---------------------------------------------------------- verdicts panel


CARDS = [
    ("EXCHANGE", "writing frees\nthe workspace",
     "prediction. a written value\nleaves the workspace\nuntil it is reused"),
    ("DUPLICATION", "the page is a\nredundant backup",
     "prediction. a written value\nis also held internally\nacross the gap"),
    ("COMPLEMENTARY", "page and workspace\nhold different content",
     "prediction. CoT values are\nabsent from the\ndirect-mode readout"),
    ("EPIPHENOMENAL", "CoT helps via extra\ncompute, not content",
     "prediction. matched-length\nfiller rescues the\nablated model"),
]


def _draw_cards(name, verdicts=None) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
    for j, (ax, (title, claim, pred)) in enumerate(zip(axes, CARDS)):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=True,
                                   facecolor="#f4f4f2", edgecolor="#cccccc",
                                   lw=1.2, zorder=1))
        ax.text(0.5, 0.90, title, ha="center", va="top", fontsize=11.5,
                fontweight="bold", zorder=3)
        ax.text(0.5, 0.78, claim, ha="center", va="top", fontsize=9.5,
                style="italic", color="#444444", zorder=3)
        ax.text(0.5, 0.575, pred, ha="center", va="top", fontsize=8.5,
                color=INK2, zorder=3)
        if verdicts:
            verdict, why, vcol = verdicts[j]
            ax.add_patch(plt.Rectangle((0.16, 0.255), 0.68, 0.105, fill=True,
                                       facecolor=vcol, edgecolor="none",
                                       zorder=2))
            ax.text(0.5, 0.307, verdict, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=3)
            ax.text(0.5, 0.215, why, ha="center", va="top", fontsize=8,
                    color=INK2, zorder=3)
    fig.suptitle("What could externalising a value mean? Four mechanisms"
                 + (", four verdicts" if verdicts else ""),
                 fontsize=13, fontweight="bold", x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, name)


def fig_hypotheses() -> None:
    """Verdict-free cards for the hypothesis-introduction slide."""
    _draw_cards("fig_hypotheses")


def fig_verdicts() -> None:
    """The same cards with the final campaign verdicts."""
    verdicts = [
        ("NOTHING TO FREE", "masked = severed in three\nmodels (p≥0.30)."
         "\nno stored copy exists", GRAY),
        ("REJECTED", "whole-transcript mask falls\nto the 10% floor."
         "\nthe value lives on the page", RED),
        ("REJECTED", "values surface internally,\nin computation order,"
         "\nin both arms (3 seeds)", RED),
        ("REJECTED", "own reasoning 98%,\nbest filler = no prefix\n"
         "(p=0.79, 3 types, 3 seeds)", RED),
    ]
    _draw_cards("fig_verdicts", verdicts)


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
    fig_datasets_accuracy(runs)
    fig_maintenance()
    fig_rescue()
    fig_cot_correspondence()
    fig_correspondence_contrast()
    fig_seed_rho()
    fig_model_probes()
    fig_filler_battery()
    fig_task_balance()
    fig_masking_schematic()
    fig_knockout()
    fig_injection()
    fig_hypotheses()
    fig_verdicts()
    table_results(runs)


if __name__ == "__main__":
    main()
