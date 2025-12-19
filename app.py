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
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return r.choices[0].message.content.strip()


def clean_list(text):
    return [a.strip() for a in text.split(",") if a.strip()]


# ============================================================
# PLANNER (AI SUGGESTS AGENTS)
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
# ROLE EXECUTION
# ============================================================

def run_role(task, role, history):
    system = f"""
You are acting as {role}.
Use professional consulting language.
No markdown, no bullets, no emojis.
Advance the work only.
"""
    user = f"TASK:\n{task}\n\nTEAM HISTORY:\n{history}"
    return llm(system, user)


def summarize(task, history):
    return llm(
        "Produce an executive summary with insights and next steps.",
        f"TASK:\n{task}\n\nFULL HISTORY:\n{history}"
    )


# ============================================================
# PDF EXPORT (PER-AGENT SECTIONS)
# ============================================================

def export_pdf(task, agent_log, summary):
    path = f"/tmp/orbita_{uuid.uuid4().hex}.pdf"
    styles = getSampleStyleSheet()

    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=10, leading=14)
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
    story.append(Paragraph(task, body))
    story.append(PageBreak())

    for agent, text in agent_log.items():
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
# AUTOMATION ENGINE (DETERMINISTIC)
# ============================================================

def run(task, agent_text):
    planner_out, suggested_agents = planner(task)

    # Agents come from editable text box
    agents = clean_list(agent_text) if agent_text.strip() else suggested_agents

    agent_log = {"Planner": planner_out}
    history = planner_out

    for agent in agents:
        out = run_role(task, agent, history)
        agent_log[agent] = out
        history += f"\n{agent}:\n{out}"

    summary = summarize(task, history)
    agent_log["Summary"] = summary

    return agent_log, summary


# ============================================================
# UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown(f"# 🧠 {BRAND_NAME}")
    gr.Markdown(TAGLINE)

    task_box = gr.Textbox(label="Task", lines=4)

    suggest_btn = gr.Button("Suggest Agents")

    agents_box = gr.Textbox(
        label="Agents to run (editable, comma-separated)",
        placeholder="AI will suggest agents here"
    )

    run_btn = gr.Button("Run ORBITA")

    log_state = gr.State()

    log_view = gr.JSON(label="Structured Agent Output")
    summary_view = gr.Textbox(label="Executive Summary", lines=10)

    pdf_btn = gr.Button("Export Full PDF")
    pdf_file = gr.File()

    def suggest_agents(task):
        _, roles = planner(task)
        return ", ".join(roles)

    def run_ui(task, agents):
        log, summary = run(task, agents)
        return log, summary, log

    suggest_btn.click(suggest_agents, task_box, agents_box)
    run_btn.click(run_ui, [task_box, agents_box], [log_view, summary_view, log_state])
    pdf_btn.click(lambda t, l, s: export_pdf(t, l, s),
                  [task_box, log_state, summary_view], pdf_file)

app.launch()
