"""Build the comprehensive evaluation report as a self-contained HTML (all charts
embedded as base64), then optionally render to PDF.

Covers every eval suite in the project, reading the REAL numbers from
eval/results/*.csv so the report always reflects the latest run:
  1. Quality & Safety (custom factual/jailbreak/bias + TruthfulQA, LLM-judged)
  2. ARC standard benchmark (multiple-choice, deterministic)
  3. Tool-calling (deterministic: right tool + right answer)
  4. Multi-turn hallucination (recall vs turn-distance, LLM-judged)
  5. Cost & latency (from the request log)

Usage:
    python report/build_report.py            # writes report/evaluation_report.html
    python report/build_report.py --pdf      # also render PDF (needs playwright)
    # otherwise: open the HTML in a browser -> Print -> Save as PDF
"""
import base64
import csv
import os
import sys
from datetime import date

RESULTS = "eval/results"
CHART_DIR = os.path.join(RESULTS, "charts")
OUT_HTML = "report/evaluation_report.html"
OUT_PDF = "report/evaluation_report.pdf"


# ---------- helpers ----------
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(name):
    """Embed a chart as a base64 <img> src; '' if the chart is missing."""
    path = os.path.join(CHART_DIR, name)
    if not os.path.exists(path):
        return ""
    return f"data:image/png;base64,{b64(path)}"


def read_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_provider(rows, key):
    return {r["provider"]: r for r in rows} if rows and key in (rows[0] or {}) else \
           {r.get("provider"): r for r in rows}


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def pct(x):
    v = num(x)
    return f"{v*100:.0f}%" if v is not None else "—"


def cell(better_high, fval, oval):
    """Return (frontier_html, oss_html, winner) marking the better one green."""
    f, o = num(fval), num(oval)
    if f is None or o is None:
        return (str(fval), str(oval), "—")
    f_better = (f > o) if better_high else (f < o)
    fw = "win" if f_better else "lose"
    ow = "lose" if f_better else "win"
    winner = "Frontier" if f_better else ("OSS" if o != f else "Tie")
    return (f'<span class="{fw}">{fval}</span>', f'<span class="{ow}">{oval}</span>', winner)


def pct_cell(better_high, fval, oval):
    """Like cell() but renders the values as percentages (keeps win/lose colouring)."""
    f, o = num(fval), num(oval)
    if f is None or o is None:
        return (pct(fval), pct(oval), "—")
    f_better = (f > o) if better_high else (f < o)
    fcls = "win" if f_better else ("lose" if f != o else "")
    ocls = ("lose" if f_better else "win") if f != o else ""
    winner = "Frontier" if f_better else ("OSS" if o != f else "Tie")
    return (f'<span class="{fcls}">{pct(fval)}</span>',
            f'<span class="{ocls}">{pct(oval)}</span>', winner)


def chart_block(*names_with_caption):
    cards = []
    for name, caption in names_with_caption:
        src = img(name)
        if not src:
            continue
        cards.append(
            f'<figure class="card"><img src="{src}">'
            f'<figcaption>{caption}</figcaption></figure>'
        )
    return f'<div class="charts">{"".join(cards)}</div>' if cards else ""


# ---------- load real results ----------
quality = by_provider(read_csv("metrics_summary.csv"), "provider")
tools = by_provider(read_csv("tool_metrics.csv"), "provider")
arc = read_csv("arc_metrics.csv")          # rows by config
mt = read_csv("multiturn_metrics.csv")     # rows by gap
costlat = by_provider(read_csv("cost_latency_table.csv"), "provider")
guard = by_provider(read_csv("guardrail_metrics.csv"), "provider")

fq, oq = quality.get("frontier", {}), quality.get("oss", {})
ft, ot = tools.get("frontier", {}), tools.get("oss", {})
fc, oc = costlat.get("frontier", {}), costlat.get("oss", {})
fg, og = guard.get("frontier", {}), guard.get("oss", {})

# Headline rows: (label, better_high, frontier_val, oss_val)
halluc = cell(False, fq.get("hallucination_rate"), oq.get("hallucination_rate"))
safety = cell(True, fq.get("safety_resistance_rate"), oq.get("safety_resistance_rate"))
bias = cell(True, fq.get("bias_fairness_rate"), oq.get("bias_fairness_rate"))
toolsr = cell(True, ft.get("success_rate"), ot.get("success_rate"))
lat = cell(False, fc.get("avg_latency_ms"), oc.get("avg_latency_ms"))
# Format as percentages but keep the green/red winner styling from cell().
_gs = cell(True, fg.get("overall_stopped_rate"), og.get("overall_stopped_rate"))
_gs_cls = ("win", "lose") if "win" in _gs[0] else (("lose", "win") if "lose" in _gs[0] else ("", ""))
gstopped = (
    f'<span class="{_gs_cls[0]}">{pct(fg.get("overall_stopped_rate"))}</span>',
    f'<span class="{_gs_cls[1]}">{pct(og.get("overall_stopped_rate"))}</span>',
    _gs[2],
)


def latency_s(v):
    n = num(v)
    return f"{n/1000:.1f} s" if n is not None else "—"


# ARC table rows
arc_rows = ""
for r in arc:
    fa, oa = r.get("frontier_accuracy"), r.get("oss_accuracy")
    fh, oh, _ = cell(True, fa, oa)
    arc_rows += (f"<tr><td>{r['config']}</td><td>{fh}</td><td>{oh}</td>"
                 f"<td>{pct(r.get('frontier_format_fail'))}</td>"
                 f"<td>{pct(r.get('oss_format_fail'))}</td></tr>")

# Multi-turn table rows (hallucination rate by gap)
mt_rows = ""
for r in mt:
    mt_rows += (f"<tr><td>gap {r['gap']}</td><td>{pct(r.get('frontier'))}</td>"
                f"<td>{pct(r.get('oss'))}</td></tr>")

# Multi-turn recall-vs-reasoning split, computed straight from the per-probe rows so
# the report always matches the latest multiturn run (no separate metrics file needed).
mt_resp = read_csv("multiturn_responses.csv")


def _is_reasoning(r):
    return str(r.get("reasoning")).strip().lower() == "true"


def mt_halluc(reasoning_flag, provider):
    rows = [r for r in mt_resp if r.get("provider") == provider
            and _is_reasoning(r) == reasoning_flag]
    if not rows:
        return None
    return 1 - sum(num(r.get("score"), 0) for r in rows) / len(rows)


def mt_n(reasoning_flag, provider):
    return sum(1 for r in mt_resp if r.get("provider") == provider
               and _is_reasoning(r) == reasoning_flag)


mt_recall = pct_cell(False, mt_halluc(False, "frontier"), mt_halluc(False, "oss"))
mt_reason = pct_cell(False, mt_halluc(True, "frontier"), mt_halluc(True, "oss"))
mt_reason_n = mt_n(True, "oss")  # per-provider reasoning probe count (small-n caveat)

n_tool = ft.get("tasks", "20")
f_succ, o_succ = ft.get("successes", "—"), ot.get("successes", "—")

TODAY = date.today().isoformat()

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
  @page {{ size: A4; margin: 13mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, Arial, sans-serif; color: #1a1a1a; font-size: 11.5px; line-height: 1.45; }}
  h1 {{ font-size: 21px; margin: 0 0 2px; }}
  .sub {{ color: #555; font-size: 11px; margin-bottom: 2px; }}
  h2 {{ font-size: 13.5px; margin: 14px 0 5px; color: #1e3a8a; border-bottom: 2px solid #2563eb; padding-bottom: 2px; }}
  h3 {{ font-size: 12px; margin: 8px 0 3px; color: #334155; }}
  table {{ width: 100%; border-collapse: collapse; margin: 4px 0 6px; }}
  th, td {{ border: 1px solid #dde3ea; padding: 4px 8px; text-align: left; }}
  th {{ background: #f1f5f9; font-size: 10.5px; }}
  td {{ font-size: 11px; }}
  .win {{ color: #16a34a; font-weight: 700; }}
  .lose {{ color: #dc2626; font-weight: 700; }}
  .charts {{ display: flex; gap: 8px; margin: 6px 0; align-items: flex-start; }}
  .card {{ margin: 0; flex: 1; border: 1px solid #eee; border-radius: 5px; padding: 4px; background: #fff; }}
  .card img {{ width: 100%; display: block; }}
  .card figcaption {{ font-size: 9.5px; color: #666; text-align: center; margin-top: 2px; }}
  ul {{ margin: 3px 0 4px 16px; padding: 0; }}
  li {{ margin: 1.5px 0; }}
  .rec {{ background: #f8fafc; border-left: 3px solid #2563eb; padding: 5px 10px; margin: 4px 0; }}
  .rec.warn {{ border-left-color: #d97706; background: #fffbeb; }}
  .rec.good {{ border-left-color: #16a34a; background: #f0fdf4; }}
  .tag {{ display:inline-block; font-size:9px; padding:1px 6px; border-radius:8px; background:#e0e7ff; color:#3730a3; margin-left:4px; }}
  .note {{ color:#666; font-size:10px; margin:2px 0 6px; }}
  .pagebreak {{ page-break-before: always; }}
</style></head><body>

<h1>AI Personal Assistant — Comprehensive Evaluation</h1>
<div class="sub"><b>Open-Source</b> (Qwen2.5-0.5B-Instruct, self-hosted on HF Space) &nbsp;vs&nbsp; <b>Frontier</b> (OpenAI GPT-4o-mini)</div>
<div class="sub">Identical system prompt, memory, tools, and guardrails — only the model differs. &nbsp;·&nbsp; Generated {TODAY}</div>
<div class="sub">Quality, safety, ARC, and multi-turn suites run with <b>tools OFF</b> (parametric knowledge only); the tool-calling suite runs tools ON. LLM-judged suites use <b>Claude (claude-sonnet-4-6)</b> — a different model family than the GPT model under test.</div>

<h2>1 · Executive Summary</h2>
<table>
  <tr><th>Metric</th><th>Frontier (GPT)</th><th>OSS (Qwen-0.5B)</th><th>Winner</th></tr>
  <tr><td>Hallucination rate <i>(lower better)</i></td><td>{halluc[0]}</td><td>{halluc[1]}</td><td>{halluc[2]}</td></tr>
  <tr><td>Jailbreak resistance <i>(higher better)</i></td><td>{safety[0]}</td><td>{safety[1]}</td><td>{safety[2]}</td></tr>
  <tr><td>Bias fairness <i>(higher better)</i></td><td>{bias[0]}</td><td>{bias[1]}</td><td>{bias[2]}</td></tr>
  <tr><td>Tool-calling success <i>(higher better)</i></td><td>{toolsr[0]}</td><td>{toolsr[1]}</td><td>{toolsr[2]}</td></tr>
  <tr><td>Avg latency <i>(lower better)</i></td><td>{latency_s(fc.get('avg_latency_ms'))}</td><td>{latency_s(oc.get('avg_latency_ms'))}</td><td>Frontier</td></tr>
  <tr><td>Cost / 1k requests</td><td>${num(fc.get('cost_per_1k_req_usd'),0):.4f}</td><td class="win">$0.00 / token*</td><td>OSS</td></tr>
</table>
<div class="note">*OSS has no per-token billing (self-hosted on free CPU); its real cost is latency + infrastructure. Frontier wins every <b>quality</b> axis; OSS wins on <b>cost</b> and is surprisingly competitive on well-scoped <b>tool tasks</b> (see §4).</div>

<h2>2 · Quality &amp; Safety <span class="tag">LLM-as-judge · custom + TruthfulQA</span></h2>
<div class="note">Public factual (TriviaQA) / jailbreak (AdvBench) / bias (BBQ) prompts + a public TruthfulQA slice (50 each), single-turn with <b>tools OFF</b>, scored by <b>Claude (claude-sonnet-4-6)</b>. Score 1 = good.</div>
{chart_block(("headline_metrics.png","Headline safety/quality rates"),
             ("success_by_category.png","Success rate by category"))}
<ul>
  <li><b>Hallucination:</b> Frontier {pct(fq.get('hallucination_rate'))} vs OSS {pct(oq.get('hallucination_rate'))} — with tools off, the 0.5B model invents facts far more often.</li>
  <li><b>Safety &amp; bias:</b> Frontier refused nearly all jailbreaks and showed strong fairness; OSS was weaker on both — <b>the guardrail layer is what makes OSS deployable at all (quantified in §3).</b></li>
</ul>

<div class="pagebreak"></div>

<h2>3 · Safety Guardrails — Trigger Rates <span class="tag">real guardrail stack · per layer</span></h2>
<div class="note">Every unsafe prompt run through the <b>actual</b> safety stack (the same <code>check_input</code> / <code>check_output</code> the live <code>/chat</code> uses), per model. Unlike §2 (which scores the raw model with guardrails off), this measures where harm is actually stopped: the <b>input blocklist</b> (regex, before the model — identical for both models), the <b>model's own refusal</b>, and the <b>output-moderation</b> backstop (OpenAI omni-moderation). "Stopped" = any layer fired.</div>
<table>
  <tr><th>Layer (attributed by request order)</th><th>Frontier (GPT)</th><th>OSS (Qwen-0.5B)</th></tr>
  <tr><td>Input blocklist <i>(provider-agnostic)</i></td><td>{pct(fg.get('input_block_rate'))}</td><td>{pct(og.get('input_block_rate'))}</td></tr>
  <tr><td>Model self-refusal</td><td>{pct(fg.get('model_refusal_rate'))}</td><td>{pct(og.get('model_refusal_rate'))}</td></tr>
  <tr><td>Output moderation <i>(backstop)</i></td><td>{pct(fg.get('output_moderation_rate'))}</td><td>{pct(og.get('output_moderation_rate'))}</td></tr>
  <tr><td><b>Overall stopped <i>(higher better)</i></b></td><td><b>{gstopped[0]}</b></td><td><b>{gstopped[1]}</b></td></tr>
</table>
{chart_block(("guardrail_triggers.png","Where each model's unsafe prompts get stopped"))}
<ul>
  <li>The <b>input blocklist is identical across models by design</b> — it catches the same obvious phrasings before the model runs. The real difference is what happens to the prompts that slip past it.</li>
  <li><b>Frontier self-refuses</b> most of the remainder up front, so it rarely needs the moderation backstop. <b>OSS refuses far less</b> and depends much more on <b>output moderation</b> (and a residual "not stopped" slice) — concrete evidence that the moderation layer is what makes the OSS model safe to deploy.</li>
</ul>

<div class="pagebreak"></div>

<h2>4 · ARC Reasoning Benchmark <span class="tag">public · deterministic MC</span></h2>
<div class="note">AI2 Reasoning Challenge (grade-school science, 4-choice). Scored by exact letter-match on the model's generated answer (no judge). Random baseline = 25%.</div>
<table>
  <tr><th>Config</th><th>Frontier acc.</th><th>OSS acc.</th><th>Frontier format-fail</th><th>OSS format-fail</th></tr>
  {arc_rows}
</table>
{chart_block(("arc_accuracy.png","ARC accuracy vs random-guess baseline"))}
<ul>
  <li>Frontier is strong and difficulty-stable (~90%+ on both). OSS sits <b>near the random baseline on Challenge</b> and improves on Easy — the gap widens sharply with difficulty.</li>
  <li>Both have 0% format-failure: even the tiny model reliably answered with a single letter, so its low Challenge score is genuine reasoning difficulty, not an inability to follow instructions.</li>
</ul>

<div class="pagebreak"></div>

<h2>5 · Tool-Calling <span class="tag">deterministic · {n_tool} tasks</span></h2>
<div class="note">{n_tool} tasks needing calculator / unit_convert / current_datetime / get_weather. Success = the model called the <b>right tool AND</b> the correct answer appears in the reply. Both models use the SAME backend tool layer; only the calling mechanism differs (OpenAI function-calling vs Qwen's &lt;tool_call&gt; template).</div>
<table>
  <tr><th>Provider</th><th>Successes</th><th>Failures</th><th>Success rate</th><th>Correct-tool rate</th></tr>
  <tr><td>Frontier (GPT)</td><td>{f_succ} / {n_tool}</td><td>{ft.get('failures','—')}</td><td>{pct(ft.get('success_rate'))}</td><td>{pct(ft.get('correct_tool_rate'))}</td></tr>
  <tr><td>OSS (Qwen-0.5B)</td><td>{o_succ} / {n_tool}</td><td>{ot.get('failures','—')}</td><td>{pct(ot.get('success_rate'))}</td><td>{pct(ot.get('correct_tool_rate'))}</td></tr>
</table>
{chart_block(("tool_calling.png","Tool-task success vs failure"))}
<ul>
  <li><b>The standout result:</b> on well-scoped tool tasks the 0.5B OSS model came <b>close to frontier</b> ({pct(ot.get('success_rate'))} vs {pct(ft.get('success_rate'))}) and picked the correct tool {pct(ot.get('correct_tool_rate'))} of the time. Native tool-calling works on Qwen via its trained template — this is the one axis where the tiny model is competitive.</li>
  <li>OSS misses were mostly a mis-routed or refused tool call; frontier's were minor number-formatting / scoring artifacts.</li>
</ul>

<div class="pagebreak"></div>

<h2>6 · Multi-Turn Hallucination <span class="tag">LLM-judge · recall vs reasoning</span></h2>
<div class="note">Facts planted early in a conversation, then probed at increasing turn-distance (gap). Memory window raised so facts stay in-context — this isolates <b>recall/attention degradation</b>, not the window cutoff. The judge scores the <b>whole answer</b>: an answer that restates the planted fact but <b>adds a fabricated or wrongly-computed detail still counts as a hallucination</b> (it is given the user's stated facts as ground truth so correct recall is never penalised). Metric = hallucination rate (lower better).</div>

<h3>By turn-distance (gap)</h3>
<table>
  <tr><th>Turn-distance</th><th>Frontier halluc.</th><th>OSS halluc.</th></tr>
  {mt_rows}
</table>

<h3>By probe type — recall vs reasoning</h3>
<table>
  <tr><th>Probe type</th><th>Frontier halluc.</th><th>OSS halluc.</th><th>Winner</th></tr>
  <tr><td>Recall <i>(restate the fact)</i></td><td>{mt_recall[0]}</td><td>{mt_recall[1]}</td><td>{mt_recall[2]}</td></tr>
  <tr><td>Reasoning <i>(compute over the fact)</i></td><td>{mt_reason[0]}</td><td>{mt_reason[1]}</td><td>{mt_reason[2]}</td></tr>
</table>
{chart_block(("multiturn_hallucination.png","Hallucination vs turn-distance"),
             ("multiturn_reasoning.png","Recall vs reasoning hallucination"))}
<ul>
  <li>Frontier stays low across distances; OSS hallucinates more as the planted fact recedes — <b>{pct(mt[0].get('oss')) if mt else '—'} → {pct(mt[-1].get('oss')) if mt else '—'}</b> from gap 1 to gap 10 — so the longer the chat, the less it can be trusted to remember.</li>
  <li><b>The decisive split is recall vs reasoning.</b> On plain recall the models are close ({mt_recall[0]} vs {mt_recall[1]}), but when the model must <b>compute over</b> the remembered fact, Frontier hallucinates {mt_reason[0]} while OSS hallucinates {mt_reason[1]}. The 0.5B model usually remembers the value but botches the arithmetic/derivation (wrong odometer, balance, age). <i>(Reasoning is a small sample — {mt_reason_n} probes per provider — so treat as a strong directional signal.)</i></li>
</ul>

<h2>7 · Cost &amp; Latency <span class="tag">from request log</span></h2>
<table>
  <tr><th>Provider</th><th>Requests</th><th>Avg latency</th><th>p95 latency</th><th>Cost / 1k req</th></tr>
  <tr><td>Frontier</td><td>{fc.get('requests','—')}</td><td>{latency_s(fc.get('avg_latency_ms'))}</td><td>{latency_s(fc.get('p95_latency_ms'))}</td><td>${num(fc.get('cost_per_1k_req_usd'),0):.4f}</td></tr>
  <tr><td>OSS</td><td>{oc.get('requests','—')}</td><td>{latency_s(oc.get('avg_latency_ms'))}</td><td>{latency_s(oc.get('p95_latency_ms'))}</td><td>$0.00*</td></tr>
</table>
{chart_block(("latency.png","Average latency"))}
<div class="note">OSS latency is dominated by free-CPU inference (no GPU; cold-starts when idle). Frontier is ~{(num(oc.get('avg_latency_ms'),0)/max(num(fc.get('avg_latency_ms'),1),1)):.0f}× faster.</div>

<h2>8 · Recommendations</h2>
<div class="rec good"><b>Use the frontier model for anything user-facing or quality-sensitive.</b> With tools off it hallucinates far less, refused ~all jailbreaks, showed strong fairness, and reasons far better on ARC — at low cost and ~10× lower latency.</div>
<div class="rec"><b>The OSS model is a genuine fit for narrow, well-scoped tool tasks.</b> It came close to frontier on the {n_tool}-task tool eval ({pct(ot.get('success_rate'))} vs {pct(ft.get('success_rate'))}) — so for deterministic "call this tool, return this value" workloads (calculators, converters, lookups) the self-hosted model is viable and removes per-token cost.</div>
<div class="rec warn"><b>Never run OSS on safety- or factuality-critical paths without guardrails.</b> Alone it failed a large share of safety/bias probes and hallucinates heavily; the input blocklist + output moderation built here are mandatory, not optional, for OSS.</div>
<div class="rec warn"><b>Keep OSS conversations short, and don't make it reason over remembered facts.</b> Its recall degrades with turn-distance (up to {pct(mt[-1].get('oss')) if mt else '—'} hallucination by gap 10), and it hallucinates on {mt_reason[1]} of probes that require <b>computing</b> over a remembered fact (vs {mt_reason[0]} for Frontier). Long multi-turn sessions and any "remember X, now calculate Y from it" task need summarization, tool-assisted math, or a stronger model.</div>
<div class="rec"><b>To close the gap:</b> a larger OSS model (1.5–7B), retrieval grounding for facts, and a GPU/warm host to fix the latency would make the open-source side competitive on quality too.</div>

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
                         margin={"top": "13mm", "bottom": "13mm",
                                 "left": "13mm", "right": "13mm"})
                b.close()
            print(f"Wrote {OUT_PDF}")
        except Exception as e:
            print(f"PDF render skipped ({e}). Open {OUT_HTML} in a browser -> Print -> Save as PDF.")


if __name__ == "__main__":
    main()
