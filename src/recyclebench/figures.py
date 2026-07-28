"""Publication figures for the thesis writeup.

Four plots, matching the reporting the exposé commits to:

1. ``rsr_forest``      -- RSR per lineage pair with 95% CIs, raw vs bundle-corrected, and the
                          |P^-| behind each estimate (an estimate on 12 behaviours must not
                          look as authoritative as one on 200).
2. ``pbr_trajectory``  -- the PBR backbone across the lineage against the pi* reference.
3. ``driver_effects``  -- RQ3 odds ratios for paraphrase classes and surface distance.
4. ``gate_operating``  -- the gate's contamination/retention trade-off across thresholds.

Style is deliberately plain: greyscale-safe, no chartjunk, CIs always shown. An estimate is
never plotted without its uncertainty, and undefined estimates are drawn as such rather than
silently dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_INK = "#222222"
_ACCENT = "#1f4e79"
_MUTED = "#9aa0a6"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_INK, labelsize=9)
    ax.grid(axis="x", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def rsr_forest(rsr_results: Sequence, out: str | Path = "results/fig_rsr_forest.png",
               min_p_minus: int = 30) -> Path:
    """Forest plot of RSR per lineage pair: raw vs bundle-corrected, with |P^-| annotated."""
    defined = [r for r in rsr_results if r.defined]
    undefined = [r for r in rsr_results if not r.defined]
    n = len(rsr_results)
    fig, ax = plt.subplots(figsize=(7.6, 1.0 + 0.85 * max(n, 1)))
    _style(ax)

    ys = list(range(len(defined)))[::-1]
    for y, r in zip(ys, defined):
        under = r.n_p_minus < min_p_minus
        # raw (muted) above corrected (accent)
        ax.plot([r.raw_ci.low, r.raw_ci.high], [y + 0.16] * 2, color=_MUTED, lw=1.6,
                solid_capstyle="butt", zorder=2)
        ax.plot(r.rsr_raw, y + 0.16, "o", ms=5, color=_MUTED, zorder=3)
        ax.plot([r.corrected_ci.low, r.corrected_ci.high], [y - 0.16] * 2,
                color=_ACCENT, lw=2.4, solid_capstyle="butt", zorder=2)
        ax.plot(r.rsr_corrected, y - 0.16, "D", ms=6, color=_ACCENT, zorder=3)
        label = f"|P⁻|={r.n_p_minus}" + ("  (under-powered)" if under else "")
        ax.text(1.005, y, label, transform=ax.get_yaxis_transform(), va="center",
                fontsize=8, color=("#b45309" if under else _INK))

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.pair[0]}\n→ {r.pair[1]}" for r in defined], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recycling Success Rate  (fraction of fixed jailbreaks re-elicited)",
                  fontsize=9, color=_INK)
    ax.axvline(0, color=_MUTED, lw=0.8)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=_MUTED, marker="o", lw=1.6, label="raw (judge-uncorrected)"),
        Line2D([], [], color=_ACCENT, marker="D", lw=2.4,
               label="bundle-corrected (headline)")],
        loc="lower right", frameon=False, fontsize=8)

    title = "Recycling across generations"
    if undefined:
        title += f"   ({len(undefined)} pair(s) undefined: P⁻ empty)"
    ax.set_title(title, fontsize=11, color=_INK, loc="left", pad=10)
    return _save(fig, out)


# ---------------------------------------------------------------------------
def pbr_trajectory(pbr_results: Sequence, out: str | Path = "results/fig_pbr.png",
                   pi_star: float = 0.15) -> Path:
    """PBR backbone across the lineage, with CIs and the pre-registered reference."""
    res = [p for p in pbr_results if p.n_resolved > 0]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    _style(ax)
    xs = list(range(len(res)))
    pts = [p.pbr_corrected for p in res]
    lo = [p.pbr_corrected - p.corrected_ci.low for p in res]
    hi = [p.corrected_ci.high - p.pbr_corrected for p in res]
    ax.errorbar(xs, pts, yerr=[lo, hi], fmt="D-", color=_ACCENT, ms=6, lw=1.8,
                capsize=4, elinewidth=1.4, zorder=3)
    ax.plot(xs, [p.pbr_raw for p in res], "o--", color=_MUTED, ms=4, lw=1.2,
            zorder=2, label="raw")
    ax.axhline(pi_star, color="#b45309", lw=1.2, ls=":",
               label=f"pre-registered reference π* = {pi_star}")
    for x, p in zip(xs, res):
        if p.n_unresolved:
            ax.annotate(f"{p.n_unresolved} unresolved", (x, p.corrected_ci.high),
                        textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=7, color=_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([p.model for p in res], rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Paraphrase Bypass Rate", fontsize=9)
    ax.set_title("Backbone: within-model paraphrase robustness along the lineage",
                 fontsize=11, color=_INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, out)


# ---------------------------------------------------------------------------
def driver_effects(driver_result, out: str | Path = "results/fig_drivers.png") -> Path:
    """RQ3: odds ratios for transformation class and surface distance (log scale)."""
    tbl = driver_result.table if hasattr(driver_result, "table") else driver_result
    rows = tbl.to_dict("records") if hasattr(tbl, "to_dict") else list(tbl)
    rows = [r for r in rows if r.get("odds_ratio") not in (None, 0)][:14]
    rows.sort(key=lambda r: r["odds_ratio"])
    fig, ax = plt.subplots(figsize=(7.6, 1.0 + 0.42 * max(len(rows), 1)))
    _style(ax)
    ys = range(len(rows))
    for y, r in zip(ys, rows):
        sig = bool(r.get("driver_after_holm"))
        ax.plot([1, r["odds_ratio"]], [y, y], color=_MUTED, lw=1.0, zorder=1)
        ax.plot(r["odds_ratio"], y, "o", ms=7 if sig else 5,
                color=_ACCENT if sig else _MUTED, zorder=3)
    ax.axvline(1.0, color=_INK, lw=1.0)
    ax.set_xscale("log")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([str(r["term"]).replace("C(transform_class)[T.", "")
                        .replace("C(category)[T.", "").replace("]", "") for r in rows],
                       fontsize=8)
    ax.set_xlabel("odds ratio for re-eliciting a patched jailbreak  (log scale)", fontsize=9)
    ax.set_title("Which reformulations recover patched jailbreaks (filled = Holm-significant)",
                 fontsize=11, color=_INK, loc="left", pad=10)
    return _save(fig, out)


# ---------------------------------------------------------------------------
def gate_operating(sensitivity_rows: Sequence[dict],
                   out: str | Path = "results/fig_gate.png") -> Path:
    """The gate's trade-off: tightening lowers contamination but discards valid variants."""
    rows = list(sensitivity_rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _style(ax)
    xs = [r["delta"] for r in rows]
    cont = [r["contamination"] for r in rows]
    ret = [r["retained_preserved"] for r in rows]
    ax.plot(xs, cont, "o-", color="#b45309", lw=1.8, ms=5,
            label="contamination  P(intent changed | admitted)")
    ax.plot(xs, ret, "D-", color=_ACCENT, lw=1.8, ms=5,
            label="retained  P(admitted | intent preserved)")
    ax.axvline(0.0, color=_INK, lw=1.0, ls="--")
    ax.annotate("frozen\noperating point", (0.0, 0.5), textcoords="offset points",
                xytext=(8, 0), fontsize=8, color=_INK, va="center")
    ax.set_xlabel("threshold shift applied to every gate component", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("Intent-gate operating curve (pre-registered sensitivity)",
                 fontsize=11, color=_INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    return _save(fig, out)


# ---------------------------------------------------------------------------
def all_figures(study, sensitivity_rows: Optional[Sequence[dict]] = None,
                outdir: str | Path = "results") -> list[Path]:
    """Render every figure a study result supports; skip what the data cannot back."""
    outdir = Path(outdir)
    made = []
    if getattr(study, "rsr", None):
        made.append(rsr_forest(study.rsr, outdir / "fig_rsr_forest.png",
                               study.cfg.decision.min_p_minus))
    if getattr(study, "pbr", None):
        made.append(pbr_trajectory(study.pbr, outdir / "fig_pbr.png",
                                   study.cfg.decision.pi_star))
    drv = getattr(study, "drivers", None)
    if drv is not None and getattr(drv, "table", None) is not None and len(drv.table):
        made.append(driver_effects(drv, outdir / "fig_drivers.png"))
    if sensitivity_rows:
        made.append(gate_operating(sensitivity_rows, outdir / "fig_gate.png"))
    return made
