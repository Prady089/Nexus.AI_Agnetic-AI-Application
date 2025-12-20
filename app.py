import os, re, uuid, datetime, asyncio, time
import nest_asyncio
nest_asyncio.apply()

import gradio as gr
from openai import OpenAI

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


# ============================================================
# BRAND CONFIG
# ============================================================

BRAND_NAME = "ORBITA"
TAGLINE = "Many Perspectives. One Insight."
AUTHOR = "Pradeep Kumar"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/prady089/"
LOGO_PATH = "orbita_logo_cropped.png"


# ============================================================
# OPENAI CONFIG
# ============================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm(system, user):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3,
    )
    return r.choices[0].message.content.strip()


def clean_list(text):
    return [a.strip() for a in text.split(",") if a.strip()]


# ============================================================
# PLANNER
# ============================================================

PLANNER_PROMPT = """
You are an expert AI orchestrator.

Create 3–6 specialist agent roles for the TASK.

Rules:
- Business-friendly role names
- Consistent naming
- No emojis or symbols

Return EXACT format:

Personas:
1. <Role> - <Why>

Workflow:
2–4 sentences.

Linear_Workflow_Roles: <Role1>, <Role2>, <Role3>
"""


def planner(task):
    out = llm(PLANNER_PROMPT, f"TASK:\n{task}")
    roles = []
    for line in out.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            roles = clean_list(line.split(":")[1])
    return out, roles


# ============================================================
# AGENT EXECUTION
# ============================================================

def run_agent(task, agent, history):
    sys = f"""
You are acting as {agent}.
Use professional consulting language.
No markdown, no bullets, no emojis.
Advance the work only.
"""
    usr = f"TASK:\n{task}\n\nHISTORY:\n{history}"
    return llm(sys, usr)


def summarize(task, history):
    return llm(
        "Produce a concise executive summary with insights and next steps.",
        f"TASK:\n{task}\n\nFULL HISTORY:\n{history}"
    )


# ============================================================
# CHAT RENDERER (WHATSAPP STYLE)
# ============================================================

def alignment(agent):
    if agent.lower().startswith("summary"):
        return "center"
    return "left" if hash(agent) % 2 == 0 else "right"


def render_chat(agent_log):
    html = """
    <style>
    .chat { font-family: Arial; }
    .bubble {
      max-width: 70%;
      padding: 12px;
      margin: 10px;
      border-radius: 12px;
      white-space: pre-wrap;
    }
    .left { background:#F1F5F9; margin-right:auto; }
    .right { background:#DBEAFE; margin-left:auto; }
    .center { background:#EDE9FE; margin:20px auto; text-align:center; }
    .agent {
      font-size:12px;
      font-weight:bold;
      cursor:pointer;
      color:#374151;
      margin-bottom:6px;
    }
    .typing { font-style:italic; color:#6B7280; }
    </style>
    <div class="chat">
    """

    for agent, text in agent_log.items():
        side = alignment(agent)
        html += f"""
        <div class="bubble {side}">
          <div class="agent">{agent}</div>
          {text}
        </div>
        """

    html += "</div>"
    return html


# ============================================================
# PDF EXPORT (UNCHANGED, STABLE)
# ============================================================

def export_pdf(task, log, summary):
    path = f"/tmp/orbita_{uuid.uuid4().hex}.pdf"
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10, leading=14)
    header = ParagraphStyle("h", parent=styles["Heading2"])

    doc = SimpleDocTemplate(path, pagesize=A4)
    story = []

    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=3*inch, height=1*inch))
    story.append(Spacer(1, 10))
    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Paragraph(f"{AUTHOR} • {AUTHOR_LINKEDIN}", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Task</b>", styles["Heading3"]))
    story.append(Paragraph(task, body))
    story.append(PageBreak())

    for agent, text in log.items():
        story.append(Paragraph(agent, header))
        for line in text.split("\n"):
            story.append(Paragraph(line, body))
        story.append(PageBreak())

    story.append(Paragraph("Executive Summary", header))
    for line in summary.split("\n"):
        story.append(Paragraph(line, body))

    doc.build(story)
    return path


# ============================================================
# CORE RUN LOGIC
# ============================================================

def run_all(task, agent_text):
    planner_out, suggested = planner(task)
    agents = clean_list(agent_text) if agent_text.strip() else suggested

    log = {"Planner": planner_out}
    history = planner_out

    for a in agents:
        log[a] = "Agent thinking..."
        yield log.copy(), ""

        out = run_agent(task, a, history)
        log[a] = out
        history += f"\n{a}:\n{out}"
        yield log.copy(), ""

    summary = summarize(task, history)
    log["Summary"] = summary
    yield log.copy(), summary


def rerun_single(task, agent, current_log):
    history = "\n".join(current_log.values())
    out = run_agent(task, agent, history)
    current_log[agent] = out
    return current_log, out


# ============================================================
# UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown(f"# 🧠 {BRAND_NAME}")
    gr.Markdown(TAGLINE)

    task_box = gr.Textbox(label="Task", lines=4)

    suggest_btn = gr.Button("Suggest Agents")
    agents_box = gr.Textbox(label="Agents (editable, comma-separated)")

    view_toggle = gr.Radio(
        ["Chat View", "Report View"],
        value="Chat View",
        label="View Mode"
    )

    run_btn = gr.Button("Run ORBITA")

    chat_view = gr.HTML()
    report_view = gr.JSON(visible=False)

    summary_box = gr.Textbox(label="Executive Summary", lines=8)
    state_log = gr.State({})

    pdf_btn = gr.Button("Export PDF")
    pdf_file = gr.File()

    def suggest(task):
        _, roles = planner(task)
        return ", ".join(roles)

    def run_stream(task, agents):
        for log, summary in run_all(task, agents):
            yield render_chat(log), log, summary

    def toggle_view(mode, log):
        if mode == "Chat View":
            return render_chat(log), gr.update(visible=True), gr.update(visible=False)
        else:
            return log, gr.update(visible=False), gr.update(visible=True)

    suggest_btn.click(suggest, task_box, agents_box)

    run_btn.click(
        run_stream,
        [task_box, agents_box],
        [chat_view, state_log, summary_box],
        stream=True
    )

    view_toggle.change(
        toggle_view,
        [view_toggle, state_log],
        [chat_view, chat_view, report_view]
    )

    pdf_btn.click(
        lambda t, l, s: export_pdf(t, l, s),
        [task_box, state_log, summary_box],
        pdf_file
    )

app.launch()
