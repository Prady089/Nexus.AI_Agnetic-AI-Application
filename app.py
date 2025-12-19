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
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle
)
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
# HELPERS
# ============================================================

def normalize_role(role: str) -> str:
    if not role:
        return ""
    role = role.replace("–", "-").replace("—", "-")
    role = re.sub(r"\s+", " ", role)
    return role.strip().lower()


def html_escape(s: str) -> str:
    if s is None:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


# ============================================================
# PLANNER AGENT (STRICT)
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
        if key and key not in seen:
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
- No markdown or decorative formatting
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
# PDF EXPORT (FULL LOG + HEADER ALIGNMENT)
# ============================================================

def generate_branded_pdf(task: str, full_log: str, summary: str, run_id: str) -> str:
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
        alignment=2,  # right
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#374151"),
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=30,
        bottomMargin=36,
    )

    story = []

    # ---------- Header: left (logo + small log id), right (author) ----------
    left_parts = []
    if os.path.exists(LOGO_PATH):
        left_parts.append(RLImage(LOGO_PATH, width=2.7 * inch, height=0.9 * inch))
    left_parts.append(Spacer(1, 2))
    left_parts.append(Paragraph(f"Log ID: {html_escape(run_id)}", small))

    right_text = (
        f"<b>{html_escape(AUTHOR)}</b><br/>"
        f"LinkedIn: {html_escape(AUTHOR_LINKEDIN)}<br/>"
        f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}"
    )
    header_table = Table(
        [[left_parts, Paragraph(right_text, right)]],
        colWidths=[doc.width * 0.55, doc.width * 0.45],
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)

    story.append(Spacer(1, 10))
    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Spacer(1, 14))

    # ---------- Task ----------
    story.append(Paragraph("<b>Task</b>", styles["Heading3"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(html_escape(task), normal))
    story.append(Spacer(1, 12))

    # ---------- Executive Summary ----------
    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))
    for line in (summary or "").split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(html_escape(line), normal))
    story.append(Spacer(1, 14))

    # ---------- Full Workflow Log (small font, FULL export) ----------
    story.append(Paragraph("<b>Workflow Log (Full)</b>", styles["Heading3"]))
    story.append(Spacer(1, 6))

    # IMPORTANT: No slicing here. This exports everything.
    for line in (full_log or "").split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(html_escape(line), small))

    doc.build(story)
    return file_path


# ============================================================
# AUTOMATION ENGINE (AUTHORITATIVE LOG)
# ============================================================

async def run_automation(task: str, selected_agents: List[str]):
    if not task.strip():
        return "", "", ""

    run_id = uuid.uuid4().hex[:10].upper()

    plan_text, suggested_agents = await planner_suggest(task)
    history = "PLANNER OUTPUT\n" + plan_text + "\n"

    # If UI passes empty (Gradio edge), default to all suggested
    if not selected_agents:
        execution_agents = suggested_agents
    else:
        selected_keys = {normalize_role(a) for a in selected_agents}
        execution_agents = [a for a in suggested_agents if normalize_role(a) in selected_keys]

    # If still empty for any reason, default to suggested
    if not execution_agents:
        execution_agents = suggested_agents

    for i, agent in enumerate(execution_agents, start=1):
        output = run_role(task, agent, history)
        history += f"\nSTEP {i}: {agent}\n{output}\n"

    summary = run_summary(task, history)
    return history, summary, run_id


def run_sync(task, agents):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_automation(task, agents))


# ============================================================
# GRADIO UI (STATE-BASED EXPORT)
# ============================================================

with gr.Blocks() as app:
    gr.Markdown(f"# 🧠 {BRAND_NAME}")
    gr.Markdown(f"### {TAGLINE}")

    task_box = gr.Textbox(
        label="Enter Task",
        lines=4,
        placeholder="Example: Curate a 4-day trip plan from Delhi to Agra focusing on the Taj Mahal."
    )

    agent_selector = gr.CheckboxGroup(
        label="Select Agents (defaults to all suggested)",
        choices=[]
    )

    plan_btn = gr.Button("Generate Agent Plan")
    run_btn = gr.Button("Run ORBITA Workflow")

    workflow_log = gr.Textbox(label="Workflow Log (UI View)", lines=18)
    final_summary = gr.Textbox(label="Executive Summary", lines=14)

    # Authoritative state for export
    workflow_state = gr.State("")
    run_id_state = gr.State("")

    pdf_btn = gr.Button("Export ORBITA PDF (Full)")
    pdf_file = gr.File(label="Download / Share PDF")

    async def populate_agents(task):
        _, agents = await planner_suggest(task)
        return gr.CheckboxGroup(choices=agents, value=agents)

    def run_and_store(task, agents):
        log, summary, run_id = run_sync(task, agents)
        return log, summary, log, run_id

    def export_pdf_action(task, full_log, summary, run_id):
        if not task or not full_log or not summary:
            return None
        return generate_branded_pdf(task, full_log, summary, run_id or "NA")

    plan_btn.click(populate_agents, inputs=task_box, outputs=agent_selector)

    run_btn.click(
        run_and_store,
        inputs=[task_box, agent_selector],
        outputs=[workflow_log, final_summary, workflow_state, run_id_state]
    )

    pdf_btn.click(
        export_pdf_action,
        inputs=[task_box, workflow_state, final_summary, run_id_state],
        outputs=pdf_file
    )

    gr.Markdown(
        f"Built by **{AUTHOR}** • "
        f"[LinkedIn]({AUTHOR_LINKEDIN}) • "
        f"© 2025 {BRAND_NAME} Labs"
    )

app.launch()
