import os
import re
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
    raise ValueError("OPENAI_API_KEY not set.")

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
# ROLE NORMALIZATION (CRITICAL)
# ============================================================

def normalize_role(role: str) -> str:
    if not role:
        return ""
    role = role.replace("–", "-").replace("—", "-")
    role = re.sub(r"\s+", " ", role)
    return role.strip().lower()


# ============================================================
# PLANNER AGENT (STRICT + STABLE)
# ============================================================

PLANNER_PROMPT = """
You are an expert orchestrator AI.

Create 3–6 specialist roles to solve the TASK.

MANDATORY RULES:
- Role names must be reused EXACTLY
- Use only ASCII hyphen "-"
- No renaming or paraphrasing later

Return output in EXACT format:

Personas:
1. <Role> - <Why needed>

Workflow:
2–4 sentences.

Linear_Workflow_Roles: <Role 1>, <Role 2>, <Role 3>
"""


async def planner_suggest(task: str) -> Tuple[str, List[str]]:
    output = run_llm(PLANNER_PROMPT, f"TASK:\n{task}")

    roles_raw = []
    for line in output.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            roles_raw = [r.strip() for r in line.split(":")[1].split(",") if r.strip()]
            break

    seen = set()
    roles = []
    for r in roles_raw:
        key = normalize_role(r)
        if key not in seen:
            seen.add(key)
            roles.append(r)

    return output, roles


# ============================================================
# ROLE AGENT (PROFESSIONAL OUTPUT)
# ============================================================

def run_role(task: str, role: str, history: str) -> str:
    system_prompt = f"""
You are acting strictly as: {role}

Writing rules:
- Professional consulting language
- No markdown, no bullets, no emojis
- Short paragraphs, executive tone

Responsibilities:
- Read TASK and TEAM HISTORY
- Do not repeat earlier content
- Advance the work
"""

    user_prompt = f"""
TASK:
{task}

TEAM HISTORY:
{history}

Proceed.
"""

    return run_llm(system_prompt, user_prompt)


# ============================================================
# SUMMARY AGENT (MANDATORY)
# ============================================================

SUMMARY_PROMPT = """
You are the Summary Agent.

Produce an EXECUTIVE SUMMARY.

Rules:
- No markdown or formatting symbols
- Clear, concise, leadership-ready

Include:
1. Task restatement
2. Key themes
3. Final insight
4. Next steps
"""


def run_summary(task: str, history: str) -> str:
    return run_llm(
        SUMMARY_PROMPT,
        f"TASK:\n{task}\n\nFULL WORKFLOW LOG:\n{history}"
    )


# ============================================================
# PDF EXPORT (USES AUTHORITATIVE STATE)
# ============================================================

def generate_branded_pdf(task: str, full_log: str, summary: str) -> str:
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

    right = ParagraphStyle(
        "right",
        parent=styles["BodyText"],
        alignment=2,
        fontSize=9,
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

    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=3 * inch, height=1 * inch))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>{AUTHOR}</b><br/>LinkedIn: {AUTHOR_LINKEDIN}<br/>"
        f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}",
        right
    ))

    story.append(Spacer(1, 12))
    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>Task</b>", styles["Heading3"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(task, normal))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))
    for line in summary.split("\n"):
        story.append(Paragraph(line, normal))

    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>Workflow Log (Condensed)</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))
    for line in full_log.split("\n")[:40]:
        story.append(Paragraph(line, small))

    doc.build(story)
    return file_path


# ============================================================
# AUTOMATION ENGINE (AUTHORITATIVE LOG)
# ============================================================

async def run_automation(task: str, selected_agents: List[str]):
    if not task.strip():
        return "", ""

    plan_text, suggested_agents = await planner_suggest(task)
    history = "PLANNER OUTPUT\n" + plan_text + "\n"

    if not selected_agents:
        execution_agents = suggested_agents
    else:
        selected_keys = {normalize_role(a) for a in selected_agents}
        execution_agents = [
            a for a in suggested_agents
            if normalize_role(a) in selected_keys
        ]

    for i, agent in enumerate(execution_agents, start=1):
        output = run_role(task, agent, history)
        history += f"\nSTEP {i}: {agent}\n{output}\n"

    summary = run_summary(task, history)
    return history, summary


def run_sync(task, agents):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_automation(task, agents))


# ============================================================
# GRADIO UI (STATE-BASED)
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
        label="Select Agents (defaults to all)",
        choices=[]
    )

    plan_btn = gr.Button("Generate Agent Plan")
    run_btn = gr.Button("Run ORBITA Workflow")

    workflow_log = gr.Textbox(label="Workflow Log", lines=18)
    final_summary = gr.Textbox(label="Executive Summary", lines=14)

    workflow_state = gr.State("")

    pdf_btn = gr.Button("Export ORBITA PDF")
    pdf_file = gr.File(label="Download / Share PDF")

    async def populate_agents(task):
        _, agents = await planner_suggest(task)
        return gr.CheckboxGroup(choices=agents, value=agents)

    def run_and_store(task, agents):
        log, summary = run_sync(task, agents)
        return log, summary, log

    def export_pdf_action(task, full_log, summary):
        if not task or not full_log or not summary:
            return None
        return generate_branded_pdf(task, full_log, summary)

    plan_btn.click(populate_agents, inputs=task_box, outputs=agent_selector)
    run_btn.click(
        run_and_store,
        inputs=[task_box, agent_selector],
        outputs=[workflow_log, final_summary, workflow_state]
    )

    pdf_btn.click(
        export_pdf_action,
        inputs=[task_box, workflow_state, final_summary],
        outputs=pdf_file
    )

    gr.Markdown(
        f"Built by **{AUTHOR}** • "
        f"[LinkedIn]({AUTHOR_LINKEDIN}) • "
        f"© 2025 {BRAND_NAME} Labs"
    )

app.launch()
