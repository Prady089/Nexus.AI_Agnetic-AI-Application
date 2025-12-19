import os, re, uuid, datetime, asyncio
import nest_asyncio
nest_asyncio.apply()

import gradio as gr
from openai import OpenAI

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
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
        temperature=0.3
    )
    return r.choices[0].message.content.strip()


def norm(x): 
    return re.sub(r"\s+", " ", x.strip()).lower()


# ============================================================
# PLANNER
# ============================================================

PLANNER_PROMPT = """
Create 3–6 specialist roles.

Rules:
- Use EXACT same role names everywhere
- No renaming
- Simple business-friendly names

Output format EXACT:

Personas:
1. <Role> - <Reason>

Workflow:
2–4 sentences.

Linear_Workflow_Roles: <Role1>, <Role2>, <Role3>
"""


def planner(task):
    out = llm(PLANNER_PROMPT, f"TASK:\n{task}")
    roles = []
    for l in out.splitlines():
        if l.startswith("Linear_Workflow_Roles:"):
            roles = [r.strip() for r in l.split(":")[1].split(",")]
    return out, roles


# ============================================================
# ROLE EXECUTION
# ============================================================

def run_role(task, role, history):
    sys = f"""
You are acting as {role}.
Write in professional consulting language.
No bullets, no markdown, no emojis.
Advance the work only.
"""
    usr = f"TASK:\n{task}\n\nHISTORY:\n{history}"
    return llm(sys, usr)


def summarize(task, history):
    return llm(
        "Produce a concise executive summary.",
        f"TASK:\n{task}\n\nFULL LOG:\n{history}"
    )


# ============================================================
# PDF (PER-AGENT SECTIONS — NEVER TRUNCATES)
# ============================================================

def export_pdf(task, run_log, summary):
    path = f"/tmp/orbita_{uuid.uuid4().hex}.pdf"
    styles = getSampleStyleSheet()

    small = ParagraphStyle("s", parent=styles["BodyText"], fontSize=9, leading=12)
    header = ParagraphStyle("h", parent=styles["Heading2"], textColor=colors.HexColor("#1F2937"))

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=40, rightMargin=40)
    story = []

    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=3*inch, height=1*inch))
    story.append(Spacer(1, 10))

    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Paragraph(f"{AUTHOR} • {AUTHOR_LINKEDIN}", small))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Task</b>", styles["Heading3"]))
    story.append(Paragraph(task, small))
    story.append(PageBreak())

    for agent, text in run_log.items():
        story.append(Paragraph(agent, header))
        for line in text.split("\n"):
            story.append(Paragraph(line, small))
        story.append(PageBreak())

    story.append(Paragraph("Executive Summary", header))
    for l in summary.split("\n"):
        story.append(Paragraph(l, small))

    doc.build(story)
    return path


# ============================================================
# AUTOMATION
# ============================================================

def run(task, agent_text):
    planner_out, suggested = planner(task)
    selected = [a.strip() for a in agent_text.split(",") if a.strip()]
    agents = selected if selected else suggested

    log = {"Planner": planner_out}
    history = planner_out

    for a in agents:
        out = run_role(task, a, history)
        log[a] = out
        history += f"\n{a}:\n{out}"

    summary = summarize(task, history)
    log["Summary"] = summary
    return log, summary


# ============================================================
# UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown(f"# 🧠 {BRAND_NAME}")
    gr.Markdown(TAGLINE)

    task = gr.Textbox(label="Task", lines=4)

    agent_input = gr.Textbox(
        label="Agents to run (comma-separated, optional)",
        placeholder="Example: Travel Planner, Budget Analyst, Local Guide"
    )

    run_btn = gr.Button("Run ORBITA")

    log_box = gr.JSON(label="Structured Workflow Output")
    summary_box = gr.Textbox(label="Executive Summary", lines=10)

    pdf_btn = gr.Button("Export Full PDF")
    pdf_file = gr.File()

    state = gr.State()

    def run_ui(t, a):
        log, summary = run(t, a)
        return log, summary, log

    run_btn.click(run_ui, [task, agent_input], [log_box, summary_box, state])

    pdf_btn.click(lambda t, l, s: export_pdf(t, l, s),
                  [task, state, summary_box], pdf_file)

app.launch()
