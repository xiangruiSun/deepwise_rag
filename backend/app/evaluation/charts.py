"""Dependency-free SVG charts for the Markdown evaluation report."""

from __future__ import annotations

from html import escape

W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 64, 24, 48, 96
INK, MUTED, GRID, ACCENT, ACCENT2 = "#1f2933", "#52606d", "#d9e2ec", "#2f6f9f", "#b5651d"


def _frame(title: str, body: str, y_label: str = "", x_label: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Helvetica, Arial, sans-serif" role="img" aria-label="{escape(title)}">'
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
        f'<text x="{PAD_L}" y="26" font-size="15" font-weight="600" fill="{INK}">{escape(title)}</text>'
        f'<text x="16" y="{PAD_T + (H - PAD_T - PAD_B) / 2}" font-size="11" fill="{MUTED}" '
        f'transform="rotate(-90 16 {PAD_T + (H - PAD_T - PAD_B) / 2})" text-anchor="middle">{escape(y_label)}</text>'
        f'<text x="{PAD_L + (W - PAD_L - PAD_R) / 2}" y="{H - 10}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">{escape(x_label)}</text>{body}</svg>'
    )


def _axis(lo: float, hi: float) -> tuple[float, float]:
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - 0.05 * span if lo < 0 else min(lo, 0.0), hi + 0.08 * span


def _y(value: float, lo: float, hi: float) -> float:
    return PAD_T + (H - PAD_T - PAD_B) * (1.0 - (value - lo) / (hi - lo))


def _grid(lo: float, hi: float, ticks: int = 5) -> str:
    out = []
    for i in range(ticks + 1):
        value = lo + (hi - lo) * i / ticks
        y = _y(value, lo, hi)
        out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="10" fill="{MUTED}" '
                   f'text-anchor="end">{value:.3g}</text>')
    return "".join(out)


def bar_chart(title: str, labels: list[str], values: list[float | None], y_label: str = "",
              signed: bool = False) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return _frame(title, f'<text x="{PAD_L}" y="{H / 2}" font-size="13" fill="{MUTED}">No completed '
                             'experiments for this chart.</text>')
    lo, hi = _axis(min([*present, 0.0]), max([*present, 0.0]))
    body = [_grid(lo, hi)]
    slot = (W - PAD_L - PAD_R) / max(len(labels), 1)
    width = min(56.0, slot * 0.68)
    zero = _y(0.0, lo, hi)
    for i, (label, value) in enumerate(zip(labels, values, strict=True)):
        cx = PAD_L + slot * (i + 0.5)
        if value is not None:
            y = _y(value, lo, hi)
            top, height = (y, zero - y) if value >= 0 else (zero, y - zero)
            color = ACCENT if (value >= 0 or not signed) else ACCENT2
            body.append(f'<rect x="{cx - width / 2:.1f}" y="{top:.1f}" width="{width:.1f}" '
                        f'height="{max(height, 0.5):.1f}" fill="{color}" rx="2"/>')
            if len(labels) <= 16:
                body.append(f'<text x="{cx:.1f}" y="{(y - 5 if value >= 0 else y + 13):.1f}" font-size="10" '
                            f'fill="{INK}" text-anchor="middle">{value:.3f}</text>')
        if len(labels) <= 24:
            ly = H - PAD_B + 14
            body.append(f'<text x="{cx:.1f}" y="{ly}" font-size="10" fill="{MUTED}" text-anchor="end" '
                        f'transform="rotate(-35 {cx:.1f} {ly})">{escape(label[:28])}</text>')
    body.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{zero:.1f}" y2="{zero:.1f}" stroke="{MUTED}"/>')
    return _frame(title, "".join(body), y_label)


def scatter_chart(title: str, points: list[tuple[str, float, float]], x_label: str, y_label: str) -> str:
    """``points``: ``(label, x, y)``; every point is directly labelled."""
    if not points:
        return _frame(title, f'<text x="{PAD_L}" y="{H / 2}" font-size="13" fill="{MUTED}">No completed '
                             'experiments for this chart.</text>')
    xs, ys = [p[1] for p in points], [p[2] for p in points]
    x_lo, x_hi = _axis(min(xs), max(xs))
    y_lo, y_hi = _axis(min(ys), max(ys))
    body = [_grid(y_lo, y_hi)]
    for i in range(6):
        value = x_lo + (x_hi - x_lo) * i / 5
        x = PAD_L + (W - PAD_L - PAD_R) * i / 5
        body.append(f'<text x="{x:.1f}" y="{H - PAD_B + 16}" font-size="10" fill="{MUTED}" '
                    f'text-anchor="middle">{value:.3g}</text>')
    for label, xv, yv in points:
        x = PAD_L + (W - PAD_L - PAD_R) * (xv - x_lo) / (x_hi - x_lo)
        y = _y(yv, y_lo, y_hi)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{ACCENT}" stroke="#ffffff" stroke-width="1.5"/>')
        body.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" font-size="10" fill="{INK}">{escape(label[:18])}</text>')
    return _frame(title, "".join(body), y_label, x_label)


def grouped_bar_chart(title: str, groups: list[str], series: dict[str, list[float | None]], y_label: str = "") -> str:
    """At most two series (baseline vs selected) so colour is never the only cue: a legend
    names both and bars sit side by side in a fixed order."""
    names = list(series)[:2]
    present = [v for n in names for v in series[n] if v is not None]
    if not present or not groups:
        return _frame(title, f'<text x="{PAD_L}" y="{H / 2}" font-size="13" fill="{MUTED}">No completed '
                             'experiments for this chart.</text>')
    lo, hi = _axis(0.0, max(present))
    body = [_grid(lo, hi)]
    slot = (W - PAD_L - PAD_R) / len(groups)
    width = min(30.0, slot * 0.34)
    colors = [ACCENT, ACCENT2]
    for gi, group in enumerate(groups):
        cx = PAD_L + slot * (gi + 0.5)
        for si, name in enumerate(names):
            value = series[name][gi]
            if value is None:
                continue
            x = cx + (si - (len(names) - 1) / 2) * (width + 3) - width / 2
            y = _y(value, lo, hi)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
                        f'height="{max(_y(0.0, lo, hi) - y, 0.5):.1f}" fill="{colors[si]}" rx="2"/>')
            body.append(f'<text x="{x + width / 2:.1f}" y="{y - 4:.1f}" font-size="9" fill="{INK}" '
                        f'text-anchor="middle">{value:.2f}</text>')
        body.append(f'<text x="{cx:.1f}" y="{H - PAD_B + 16}" font-size="10" fill="{MUTED}" '
                    f'text-anchor="middle">{escape(group[:26])}</text>')
    for si, name in enumerate(names):
        x = PAD_L + si * 220
        body.append(f'<rect x="{x}" y="{H - 44}" width="12" height="12" fill="{colors[si]}" rx="2"/>')
        body.append(f'<text x="{x + 18}" y="{H - 34}" font-size="11" fill="{INK}">{escape(name[:30])}</text>')
    return _frame(title, "".join(body), y_label)
