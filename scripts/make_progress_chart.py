"""Draw assets/progress.svg: public leaderboard score per probe, with the public references.

    python scripts/make_progress_chart.py && rsvg-convert -w 1600 assets/progress.svg -o assets/progress.png

Scores come from docs/experiments/LOG.md. Add a probe by adding a row to PROBES.
"""

from pathlib import Path

PROBES = [  # (label, public LB, short note)
    ("1", 0.626, "spine + SVC"),
    ("2", 0.629, "half Sinkhorn"),
    ("3", 0.631, "+ seeds"),
    ("4", 0.627, "+ in-style"),
    ("5", 0.631, "+ ATLAS"),
    ("6", 0.622, "peer corr."),
    ("7", 0.631, "+ our LoRA"),
    ("8", 0.631, "Gauss 0.4"),
    ("9", 0.637, "Gauss 1.0"),
    ("10", 0.630, "14B Gauss"),
]
REFS = [(0.620, "Nishant Dahal's published blend"), (0.639, "best public notebook")]
LO, HI = 0.615, 0.643
W, H, L, R, T, B = 1200, 520, 90, 40, 80, 110


def y(v):
    return T + (HI - v) / (HI - LO) * (H - T - B)


def main():
    step = (W - L - R) / len(PROBES)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           '<text x="40" y="44" font-size="24" font-weight="bold" fill="#111">Public leaderboard by probe</text>']
    for v in (0.620, 0.625, 0.630, 0.635, 0.640):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{L - 10}" y="{y(v) + 5:.1f}" font-size="14" fill="#666" text-anchor="end">{v:.3f}</text>')
    for v, name in REFS:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#999" stroke-dasharray="6 5"/>')
        out.append(f'<text x="{L + 8}" y="{y(v) - 7:.1f}" font-size="14" fill="#777" text-anchor="start">{name} {v:.3f}</text>')
    pts = [(L + step * (i + 0.5), y(v)) for i, (_, v, _) in enumerate(PROBES)]
    out.append('<polyline fill="none" stroke="#2a62b8" stroke-width="3" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
    for (a, b), (label, v, note) in zip(pts, PROBES):
        out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" fill="#2a62b8"/>')
        out.append(f'<text x="{a:.1f}" y="{b - 16:.1f}" font-size="16" font-weight="bold" fill="#111" text-anchor="middle">{v:.3f}</text>')
        out.append(f'<text x="{a:.1f}" y="{H - B + 30}" font-size="16" fill="#111" text-anchor="middle">Probe {label}</text>')
        out.append(f'<text x="{a:.1f}" y="{H - B + 52}" font-size="13" fill="#666" text-anchor="middle">{note}</text>')
    out.append("</svg>")
    Path("assets/progress.svg").write_text("\n".join(out))


if __name__ == "__main__":
    main()
