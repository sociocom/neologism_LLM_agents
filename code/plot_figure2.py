"""Regenerate the manuscript's Figure 2 forest plot from analysis output.

Run `python code/reproduce_results.py` first. The plotted coefficients,
confidence limits, nominal p values and adjusted p values are read from its
CSV outputs rather than written into this file.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def font_names() -> tuple[str, str, str]:
    candidates = [
        (Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
         Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
         Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("FigureRegular", str(regular)))
            pdfmetrics.registerFont(TTFont("FigureBold", str(bold)))
            return "FigureRegular", "FigureBold", "×"
    return "Helvetica", "Helvetica-Bold", "x"


def result(model: str, term: str, standardized: bool,
           coefficients: pd.DataFrame, tests: pd.DataFrame) -> dict:
    row = coefficients.loc[coefficients.model.eq(model) & coefficients.term.eq(term)]
    test = tests.loc[tests.model.eq(model) & tests.term.eq(term)]
    assert len(row) == len(test) == 1, (model, term)
    row, test = row.iloc[0], test.iloc[0]
    if standardized:
        estimate, lower, upper = row.beta_std, row.beta_std_ci_low, row.beta_std_ci_high
    else:
        estimate, lower, upper = row.b, row.ci_low, row.ci_high
    return {"estimate": float(estimate), "lower": float(lower), "upper": float(upper),
            "p": float(test.p), "adjusted": float(test.p_adjusted)}


def build() -> Path:
    coefficients = pd.read_csv(RESULTS / "model_coefficients.csv")
    tests = pd.read_csv(RESULTS / "focal_tests.csv")
    regular, bold, multiply = font_names()
    BLUE = colors.HexColor("#155E9B")
    DARK = colors.HexColor("#444444")
    INK = colors.HexColor("#222222")
    RULE = colors.HexColor("#BBBBBB")
    MUTED = colors.HexColor("#5E5E5E")
    width, height = 920, 440
    output = RESULTS / "figure2.pdf"
    c = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    c.setTitle("Figure 2. Between-wave contrasts and exploratory associations")
    c.translate(0, -35)
    plot_left, plot_right = 310, 555

    def txt(x: float, y: float, label: str, size=10, heavy=False, color=INK):
        c.setFont(bold if heavy else regular, size)
        c.setFillColor(color)
        c.drawString(x, y, label)

    def right(x: float, y: float, label: str, size=10, heavy=False, color=INK):
        c.setFont(bold if heavy else regular, size)
        c.setFillColor(color)
        c.drawRightString(x, y, label)

    def rule(y: float):
        c.setStrokeColor(RULE)
        c.setLineWidth(.6)
        c.line(38, y, 895, y)

    def xpos(value: float, domain: tuple[float, float]) -> float:
        return plot_left + (value - domain[0]) / (domain[1] - domain[0]) * (plot_right - plot_left)

    def reference(domain: tuple[float, float], bottom: float, top: float):
        c.setStrokeColor(colors.HexColor("#888888"))
        c.setLineWidth(.85)
        c.setDash(2, 2.5)
        x = xpos(0, domain)
        c.line(x, bottom, x, top)
        c.setDash()

    def axis(y: float, domain: tuple[float, float], ticks: list[float], label: str):
        c.setStrokeColor(RULE)
        c.setLineWidth(.7)
        c.line(plot_left, y, plot_right, y)
        for value in ticks:
            x = xpos(value, domain)
            c.line(x, y, x, y - 4)
            c.setFont(regular, 9.4)
            c.setFillColor(MUTED)
            c.drawCentredString(x, y - 15, f"{value:g}")
        txt(plot_left, y - 25, label, 9.4, color=MUTED)

    def fmt(value: float) -> str:
        return f"{value:.3f}"

    def p_fmt(value: float) -> str:
        return "<.001" if value < .001 else f"{value:.3f}".removeprefix("0")

    def estimate(y: float, label: str, values: dict, domain: tuple[float, float],
                 show_adjusted=True):
        color = BLUE if values["adjusted"] < .05 else DARK
        txt(42, y - 5, label, 12.1, color=color)
        lo, point, hi = (xpos(values[k], domain) for k in ("lower", "estimate", "upper"))
        c.setStrokeColor(color)
        c.setLineWidth(1.7)
        c.line(lo, y, hi, y)
        for x in (lo, hi):
            c.line(x, y - 5.3, x, y + 5.3)
        c.setFillColor(color)
        c.circle(point, y, 4.1, fill=1, stroke=0)
        display = f'{fmt(values["estimate"])} [{fmt(values["lower"])}, {fmt(values["upper"])}]'
        right(735, y - 4, display, 12.4, color=color)
        right(805, y - 4, p_fmt(values["p"]), 12.4, color=color)
        right(895, y - 4, p_fmt(values["adjusted"]) if show_adjusted else "-", 12.4, color=color)

    def heading(y: float, letter: str, title: str, subtitle: str):
        txt(39, y, letter, 16.5, heavy=True)
        txt(68, y, title, 16.0, heavy=True)
        txt(62, y - 16, subtitle, 8.5, color=MUTED)

    right(735, 448, "Estimate (95% CI)", 12.4, heavy=True)
    right(805, 448, "Nominal p", 12.4, heavy=True)
    right(895, 448, "BH-FDR p", 12.4, heavy=True)
    rule(439)
    heading(417, "A", "Between-wave contrast (N = 100)",
            "Outcome: post-discussion four-item identity mean, adjusted for midpoint identity")
    reference((-.4, .4), 354, 382)
    estimate(368, "Proposal vs. comparison wave",
             result("primary_ancova", "proposal_wave", False, coefficients, tests), (-.4, .4), False)
    axis(346, (-.4, .4), [-.4, 0, .4], "Identity-score points (b)")
    rule(315)
    heading(295, "B", "Differences in associations between waves (N = 100)",
            "Outcome: change in the four-item identity sum (separate exploratory models)")
    reference((-.4, .7), 198, 270)
    estimate(264, f"Wave {multiply} message count",
             result("wave_message_change", "proposal_wave:total_message_count_c", True,
                    coefficients, tests), (-.4, .7))
    estimate(234, f"Wave {multiply} team impression",
             result("wave_impression_change", "proposal_wave:team_impression_end_score_c", True,
                    coefficients, tests), (-.4, .7))
    estimate(204, f"Wave {multiply} message count {multiply} team impression",
             result("wave_message_impression_change",
                    "proposal_wave:total_message_count_c:team_impression_end_score_c",
                    True, coefficients, tests), (-.4, .7))
    axis(186, (-.4, .7), [-.4, 0, .4, .7], "Standardized coefficient (beta)")
    rule(157)
    heading(137, "C", "Association within the proposal wave (N = 50)",
            "Outcome: change in the four-item identity sum")
    reference((-.4, .7), 93, 115)
    estimate(104, "Neologism evaluation",
             result("word_evaluation", "neologism_evaluation_sum", True,
                    coefficients, tests), (-.4, .7))
    axis(85, (-.4, .7), [-.4, 0, .4, .7], "Standardized coefficient (beta)")
    c.showPage()
    c.save()
    return output


if __name__ == "__main__":
    print(build())
