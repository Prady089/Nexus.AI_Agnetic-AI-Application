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
You are an expert orchestrator AI that designs teams of specialist agents.

Given a TASK:
1. Identify what type of problem this is.
2. Propose 3–7 personas needed.
3. Provide a short justification for each.
4. Provide a LINEAR workflow order.

Follow EXACT structure:

Personas:
1. <Role> - <Why needed>

Workflow:
2–4 sentence description.

Linear_Workflow_Roles: <Role 1>, <Role 2>, <Role 3>, ...
"""


async def planner_suggest(task: str) -> Tuple[str, List[str]]:
    output = run_llm(
        PLANNER_PROMPT,
        f"TASK:\n{task}\n\nFollow the required output structure."
    )

    roles = []
    for line in output.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            roles = [r.strip() for r in line.split(":")[1].split(",") if r.strip()]
            break

    return output, roles


# ============================================================
# ROLE EXECUTION
# ============================================================

def run_role(task: str, role: str, history: str) -> str:
    system_prompt = f"""
You are acting strictly as: {role}

Rules:
- Contribute only from this role’s perspective
- Read TASK and TEAM HISTORY
- Do NOT repeat earlier content
- Advance the work
- Use headings and bullet points
"""

    user_prompt = f"""
TASK:
{task}

TEAM HISTORY:
{history}

Now act as {role} and advance the work.
"""

    return run_llm(system_prompt, user_prompt)


# ============================================================
# SUMMARY AGENT (MANDATORY)
# ============================================================

SUMMARY_PROMPT = """
You are the Summary Agent.

Summarize the entire multi-agent workflow.

Your output MUST include:
- Task overview
- Key contribution from each agent
- Final synthesized insight
- Clear next steps or recommendations
"""


def run_summary(task: str, history: str) -> str:
    return run_llm(
        SUMMARY_PROMPT,
        f"TASK:\n{task}\n\nFULL WORKFLOW LOG:\n{history}"
    )


# ============================================================
# PDF EXPORT (ORBITA BRANDED)
# ============================================================

def generate_branded_pdf(workflow_log: str, summary: str) -> str:
    file_path = f"/tmp/orbita_workflow_{uuid.uuid4().hex}.pdf"

    styles = getSampleStyleSheet()
    small = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#374151"),
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        topMargin=36,
        bottomMargin=36,
        leftMargin=48,
        rightMargin=48,
    )

    story = []

    # Logo
    if os.path.exists(LOGO_PATH):
        logo_w = 3.5 * inch
        story.append(RLImage(LOGO_PATH, width=logo_w, height=logo_w * 0.35))
        story.append(Spacer(1, 10))
    else:
        story.append(Paragraph(BRAND_NAME, styles["Title"]))

    # Header
    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Built by {AUTHOR} • LinkedIn: {AUTHOR_LINKEDIN}",
        small
    ))
    story.append(Paragraph(
        f"Generated on: {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}",
        small
    ))
    story.append(Spacer(1, 14))

    # Workflow
    story.append(Paragraph("Workflow Log", styles["Heading2"]))
    story.append(Spacer(1, 6))
    for line in workflow_log.split("\n"):
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, styles["BodyText"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Final Summary", styles["Heading2"]))
    story.append(Spacer(1, 6))
    for line in summary.split("\n"):
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe, styles["BodyText"]))

    doc.build(story)
    return file_path


# ============================================================
# AUTOMATION ENGINE
# ============================================================

async def run_automation(task: str, selected_agents: List[str]):
    if not task.strip():
        return "⚠️ Please enter a task.", ""

    plan_text, suggested_agents = await planner_suggest(task)
    history = "=== PLANNER OUTPUT ===\n" + plan_text + "\n"

    execution_agents = [a for a in suggested_agents if a in selected_agents]
    if not execution_agents:
        return history + "\n❌ No agents selected.", ""

    for i, agent in enumerate(execution_agents, start=1):
        output = run_role(task, agent, history)
        history += f"\n\n=== STEP {i}: {agent} ===\n{output}\n"

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
        placeholder="Example: Design an AI-powered consulting assistant."
    )

    agent_selector = gr.CheckboxGroup(
        label="Select Agents (Summary Agent runs automatically)",
        choices=[]
    )

    plan_btn = gr.Button("🧠 Generate Agent Plan")
    run_btn = gr.Button("🚀 Run ORBITA Workflow")

    workflow_log = gr.Textbox(label="Workflow Log", lines=22)
    final_summary = gr.Textbox(label="Final Insight (Summary Agent)", lines=14)

    pdf_btn = gr.Button("📄 Export ORBITA PDF")
    pdf_file = gr.File(label="Download / Share PDF")

    async def populate_agents(task):
        _, agents = await planner_suggest(task)
        return gr.CheckboxGroup(choices=agents, value=agents)

    def export_pdf_action(log, summary):
        if not log or not summary:
            return None
        return generate_branded_pdf(log, summary)

    plan_btn.click(populate_agents, inputs=task_box, outputs=agent_selector)
    run_btn.click(run_sync, inputs=[task_box, agent_selector], outputs=[workflow_log, final_summary])
    pdf_btn.click(export_pdf_action, inputs=[workflow_log, final_summary], outputs=pdf_file)

    gr.Markdown(
        f"Built by **{AUTHOR}** • "
        f"[LinkedIn]({AUTHOR_LINKEDIN}) • "
        f"© 2025 {BRAND_NAME} Labs"
    )

app.launch()
