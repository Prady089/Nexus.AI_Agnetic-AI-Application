import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
import uuid
import datetime
from typing import List, Tuple

import gradio as gr
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


# ============================================================
# ORBITA BRAND CONFIG
# ============================================================

BRAND_NAME = "ORBITA"
TAGLINE = "Many Perspectives. One Insight."
AUTHOR = "Pradeep Kumar"
AUTHOR_LINKEDIN = "https://www.linkedin.com/in/prady089/"
LOGO_PATH = "orbita_logo_cropped.png"


# ============================================================
# OPENAI CONFIG
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment.")

client = OpenAI(api_key=OPENAI_API_KEY)


def run_llm(system_prompt: str, user_prompt: str, model="gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# PLANNER AGENT
# ============================================================

PLANNER_PROMPT = """
You are an expert orchestrator AI.

Your job is to design a professional multi-agent workflow.

Instructions:
- Identify the nature of the task
- Propose 3–6 specialist roles
- Keep role names business-friendly
- Provide a logical execution order

Required output format (exact):

Personas:
1. <Role> – <Why this role is needed>

Workflow:
2–4 sentences explaining the overall approach.

Linear_Workflow_Roles: <Role 1>, <Role 2>, <Role 3>
"""


async def planner_suggest(task: str) -> Tuple[str, List[str]]:
    output = run_llm(
        PLANNER_PROMPT,
        f"TASK:\n{task}"
    )

    roles = []
    for line in output.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            roles = [r.strip() for r in line.split(":")[1].split(",") if r.strip()]
            break

    return output, roles


# ============================================================
# ROLE AGENT (PROFESSIONAL OUTPUT)
# ============================================================

def run_role(task: str, role: str, history: str) -> str:
    system_prompt = f"""
You are acting strictly as: {role}

Writing rules (MANDATORY):
- Professional consulting tone
- No markdown symbols, no bullets, no emojis
- No headings like ### or **
- Use short, clear paragraphs
- Be concise and executive-friendly

Responsibilities:
- Read the TASK and TEAM HISTORY
- Do not repeat earlier content
- Advance the analysis from your role’s perspective
"""

    user_prompt = f"""
TASK:
{task}

TEAM HISTORY:
{history}

Proceed with your contribution.
"""

    return run_llm(system_prompt, user_prompt)


# ============================================================
# SUMMARY AGENT (MANDATORY)
# ============================================================

SUMMARY_PROMPT = """
You are the Summary Agent.

Produce a PROFESSIONAL EXECUTIVE SUMMARY.

Rules:
- No markdown formatting
- No symbols or decorative elements
- Clear, structured, leadership-ready language

Your summary must include:
1. Restatement of the original task
2. Key themes discussed by agents
3. Final synthesized insight
4. Practical next steps
"""


def run_summary(task: str, history: str) -> str:
    return run_llm(
        SUMMARY_PROMPT,
        f"TASK:\n{task}\n\nFULL WORKFLOW LOG:\n{history}"
    )


# ============================================================
# PDF EXPORT (EXECUTIVE-READY)
# ============================================================

def generate_branded_pdf(task: str, workflow_log: str, summary: str) -> str:
    file_path = f"/tmp/orbita_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()

    small = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#6B7280"),
    )

    normal = ParagraphStyle(
        "normal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
    )

    right_align = ParagraphStyle(
        "right",
        parent=styles["BodyText"],
        fontSize=9,
        alignment=2,
        textColor=colors.HexColor("#374151"),
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    story = []

    # Header
    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=3 * inch, height=1 * inch))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>{AUTHOR}</b><br/>LinkedIn: {AUTHOR_LINKEDIN}<br/>"
        f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}",
        right_align
    ))

    story.append(Spacer(1, 10))
    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Spacer(1, 16))

    # Task
    story.append(Paragraph("<b>Task</b>", styles["Heading3"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(task, normal))
    story.append(Spacer(1, 14))

    # Summary
    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))
    for line in summary.split("\n"):
        story.append(Paragraph(line, normal))

    story.append(Spacer(1, 16))

    # Condensed Workflow Log
    story.append(Paragraph("<b>Workflow Log (Condensed)</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))
    for line in workflow_log.split("\n")[:25]:
        story.append(Paragraph(line, small))

    doc.build(story)
    return file_path


# ============================================================
# AUTOMATION ENGINE
# ============================================================

async def run_automation(task: str, selected_agents: List[str]):
    if not task.strip():
        return "Please enter a task.", ""

    plan_text, suggested_agents = await planner_suggest(task)
    history = "PLANNER OUTPUT\n" + plan_text + "\n"

    # If user didn't explicitly select agents, default to all suggested
    if not selected_agents:
        execution_agents = suggested_agents
    else:
        execution_agents = [a for a in suggested_agents if a in selected_agents]

    if not execution_agents:
        return history + "\nNo agents available to run.", ""

    for i, agent in enumerate(execution_agents, start=1):
        output = run_role(task, agent, history)
        history += f"\nSTEP {i}: {agent}\n{output}\n"

    summary = run_summary(task, history)
    return history, summary


def run_sync(task, agents):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_automation(task, agents))


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown(f"# 🧠 {BRAND_NAME}")
    gr.Markdown(f"### {TAGLINE}")

    task_box = gr.Textbox(
        label="Enter Task",
        lines=4,
        placeholder="Example: Enhance executive Power BI dashboards using AI and agentic AI."
    )

    agent_selector = gr.CheckboxGroup(
        label="Select Agents (Summary Agent runs automatically)",
        choices=[]
    )

    plan_btn = gr.Button("Generate Agent Plan")
    run_btn = gr.Button("Run ORBITA Workflow")

    workflow_log = gr.Textbox(label="Workflow Log", lines=18)
    final_summary = gr.Textbox(label="Executive Summary", lines=14)

    pdf_btn = gr.Button("Export ORBITA PDF")
    pdf_file = gr.File(label="Download / Share PDF")

    async def populate_agents(task):
        _, agents = await planner_suggest(task)
        return gr.CheckboxGroup(choices=agents, value=agents)

    def export_pdf_action(task, log, summary):
        if not task or not log or not summary:
            return None
        return generate_branded_pdf(task, log, summary)

    plan_btn.click(populate_agents, inputs=task_box, outputs=agent_selector)
    run_btn.click(run_sync, inputs=[task_box, agent_selector], outputs=[workflow_log, final_summary])
    pdf_btn.click(export_pdf_action, inputs=[task_box, workflow_log, final_summary], outputs=pdf_file)

    gr.Markdown(
        f"Built by **{AUTHOR}** • "
        f"[LinkedIn]({AUTHOR_LINKEDIN}) • "
        f"© 2025 {BRAND_NAME} Labs"
    )

app.launch()
