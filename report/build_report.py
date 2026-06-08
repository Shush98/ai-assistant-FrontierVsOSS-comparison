"""Build the 1-page evaluation report as a self-contained HTML (charts embedded
as base64), then render to PDF.

Usage:
    python report/build_report.py            # writes report/evaluation_report.html
    # then render to PDF (one of):
    #   - open the HTML in a browser -> Print -> Save as PDF
    #   - or: pip install playwright && playwright install chromium
    #         python report/build_report.py --pdf
"""
import base64
import os
import sys

CHART_DIR = "eval/results/charts"
OUT_HTML = "report/evaluation_report.html"
OUT_PDF = "report/evaluation_report.pdf"


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(name):
    return f"data:image/png;base64,{b64(os.path.join(CHART_DIR, name))}"


HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 14mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, Arial, sans-serif; color: #1a1a1a; font-size: 12px; line-height: 1.4; }}
  h1 {{ font-size: 20px; margin: 0 0 2px; }}
  .sub {{ color: #666; font-size: 11px; margin-bottom: 10px; }}
  h2 {{ font-size: 13px; margin: 12px 0 5px; border-bottom: 2px solid #2563eb; padding-bottom: 2px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 4px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; }}
  th {{ background: #f1f5f9; }}
  .win {{ color: #16a34a; font-weight: 600; }}
  .lose {{ color: #dc2626; font-weight: 600; }}
  .charts {{ display: flex; gap: 8px; margin: 6px 0; }}
  .charts img {{ width: 33%; border: 1px solid #eee; border-radius: 4px; }}
  ul {{ margin: 4px 0 4px 18px; padding: 0; }}
  li {{ margin: 2px 0; }}
  .rec {{ background: #f8fafc; border-left: 3px solid #2563eb; padding: 6px 10px; margin: 4px 0; }}
</style></head><body>

<h1>AI Personal Assistant — Evaluation Report</h1>
<div class="sub">Open-Source (Qwen2.5-0.5B-Instruct, self-hosted) vs Frontier (OpenAI GPT) &nbsp;·&nbsp; LLM-as-judge + TruthfulQA</div>

<h2>Headline Results</h2>
<table>
  <tr><th>Metric</th><th>Frontier (GPT)</th><th>Open Source (Qwen-0.5B)</th><th>Winner</th></tr>
  <tr><td>Hallucination rate <i>(lower = better)</i></td><td class="win">0.14</td><td class="lose">0.68</td><td>Frontier</td></tr>
  <tr><td>Jailbreak resistance <i>(higher = better)</i></td><td class="win">1.00</td><td class="lose">0.50</td><td>Frontier</td></tr>
  <tr><td>Bias fairness <i>(higher = better)</i></td><td class="win">1.00</td><td class="lose">0.50</td><td>Frontier</td></tr>
  <tr><td>Avg latency <i>(lower = better)</i></td><td class="win">~2.8 s</td><td class="lose">~16.5 s</td><td>Frontier</td></tr>
  <tr><td>Cost per 1k requests</td><td>~$0.057</td><td class="win">$0.00 / token*</td><td>OSS</td></tr>
</table>
<div class="sub">*OSS has no per-token cost (self-hosted on free HF Space CPU); its cost is infrastructure + latency, not per-call billing.</div>

<h2>Infographics</h2>
<div class="charts">
  <img src="{img('headline_metrics.png')}">
  <img src="{img('success_by_category.png')}">
  <img src="{img('latency.png')}">
</div>

<h2>Methodology</h2>
<ul>
  <li><b>Datasets:</b> custom factual, jailbreak, and bias prompts + a 20-item slice of the public <b>TruthfulQA</b> benchmark.</li>
  <li><b>Procedure:</b> each prompt run single-turn through both models with identical system prompt and config; only the model differs.</li>
  <li><b>Scoring:</b> LLM-as-judge (GPT-4o, temperature 0) with a strict per-category rubric — factual correctness, refusal of harmful requests, and fairness. Score 1 = good.</li>
</ul>

<h2>Recommendations</h2>
<div class="rec"><b>Use the frontier model for any user-facing assistant.</b> It hallucinates ~5× less, refused 100% of jailbreaks, and showed no measurable bias — at low cost and 6× lower latency.</div>
<div class="rec"><b>The OSS model is viable only with a guardrail layer.</b> Alone it failed half of safety and bias probes; the input/output guardrails built here are what make it deployable.</div>
<div class="rec"><b>For cost-sensitive, low-stakes, high-volume tasks</b> (e.g. classification, drafting), the self-hosted OSS model removes per-token cost — but invest in a larger OSS model, retrieval grounding, and stronger safety filtering to close the quality gap.</div>

</body></html>"""


def main():
    os.makedirs("report", exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(HTML)
    print(f"Wrote {OUT_HTML}")

    if "--pdf" in sys.argv:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b = p.chromium.launch()
                page = b.new_page()
                page.goto("file://" + os.path.abspath(OUT_HTML))
                page.pdf(path=OUT_PDF, format="A4",
                         margin={"top": "14mm", "bottom": "14mm",
                                 "left": "14mm", "right": "14mm"})
                b.close()
            print(f"Wrote {OUT_PDF}")
        except Exception as e:
            print(f"PDF render skipped ({e}). Open {OUT_HTML} in a browser -> Print -> Save as PDF.")


if __name__ == "__main__":
    main()
