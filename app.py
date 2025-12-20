import os, uuid, datetime, time
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


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
- No emojis or symbols
- Consistent naming

Return EXACT format:

Personas:
1. <Role> - <Why>

Workflow:
2–4 sentences.

Linear_Workflow_Roles: <Role1>, <Role2>, <Role3>
"""


def planner(task):
    output = llm(PLANNER_PROMPT, f"TASK:\n{task}")
    roles = []
    for line in output.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            roles = clean_list(line.split(":")[1])
    return output, roles


# ============================================================
# AGENT EXECUTION
# ============================================================

def run_agent(task, agent, history):
    system = f"""
You are acting as {agent}.
Write in professional consulting language.
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
# CHAT RENDERER (FIXED COLORS + ALIGNMENT)
# ============================================================

def alignment(agent, index):
    if agent.lower().startswith("summary"):
        return "center"
    return "left" if index % 2 == 0 else "right"


def render_chat(agent_log):
    html = """
    <style>
    .chat {
      font-family: Arial, sans-serif;
    }
    .bubble {
      max-width: 70%;
      padding: 14px;
      margin: 10px;
      border-radius: 14px;
      white-space: pre-wrap;
      color: #111827;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }
    .left {
      background: #E5E7EB;   /* Gradio grey */
      margin-right: auto;
    }
    .right {
      background: #FEF3C7;   /* Gradio orange tint */
      margin-left: auto;
    }
    .center {
      background: #EDE9FE;
      margin: 20px auto;
      text-align: center;
      font-weight: 600;
    }
    .agent {
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
      color: #F97316;        /* Gradio orange */
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .typing {
      font-style: italic;
      color: #6B7280;
    }
    </style>
    <div class="chat">
    """

    for idx, (agent, text) in enumerate(agent_log.items()):
        side = alignment(agent, idx)
        html += f"""
        <div class="bubble {side}">
          <div class="agent">{agent}</div>
          {text}
        </div>
        """

    html += "</div>"
    return html


# ============================================================
# PDF EXPORT (FULL, SAFE)
# ============================================================

def export_pdf(task, log, summary):
    path = f"/tmp/orbita_{uuid.uuid4().hex}.pdf"
    styles = getSampleStyleSheet()

    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    header = ParagraphStyle("header", parent=styles["Heading2"])

    doc = SimpleDocTemplate(path, pagesize=A4)
    story = []

    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=3 * inch, height=1 * inch))
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
# CORE WORKFLOW (GENERATOR FOR STREAMING)
# ============================================================

def run_all(task, agent_text):
    planner_out, suggested_agents = planner(task)
    agents = clean_list(agent_text) if agent_text.strip() else suggested_agents

    log = {"Planner": planner_out}
    history = planner_out

    for agent in agents:
        log[agent] = "<span class='typing'>Agent thinking...</span>"
        yield render_chat(log), log, ""

        time.sleep(0.25)

        output = run_agent(task, agent, history)
        log[agent] = output
        history += f"\n{agent}:\n{output}"
        yield render_chat(log), log, ""

    summary = summarize(task, history)
    log["Summary"] = summary
    yield render_chat(log), log, summary


# ============================================================
# UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown(f"# 🧠 {BRAND_NAME}")
    gr.Markdown(TAGLINE)

    task_box = gr.Textbox(label="Task", lines=4)

    suggest_btn = gr.Button("Suggest Agents")
    agents_box = gr.Textbox(
        label="Agents (editable, comma-separated)",
        placeholder="AI will suggest agents here"
    )

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

    def suggest_agents(task):
        _, roles = planner(task)
        return ", ".join(roles)

    def toggle_view(mode, log):
        if mode == "Chat View":
            return render_chat(log), gr.update(visible=True), gr.update(visible=False)
        else:
            return log, gr.update(visible=False), gr.update(visible=True)

    suggest_btn.click(suggest_agents, task_box, agents_box)

    run_btn.click(
        run_all,
        inputs=[task_box, agents_box],
        outputs=[chat_view, state_log, summary_box]
    )

    view_toggle.change(
        toggle_view,
        inputs=[view_toggle, state_log],
        outputs=[chat_view, chat_view, report_view]
    )

    pdf_btn.click(
        lambda t, l, s: export_pdf(t, l, s),
        inputs=[task_box, state_log, summary_box],
        outputs=pdf_file
    )

app.queue()
app.launch()
