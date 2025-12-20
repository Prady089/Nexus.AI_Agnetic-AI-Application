import os, uuid, time, re
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
        temperature=0.25,
    )
    return response.choices[0].message.content.strip()


def clean_list(text):
    return [a.strip() for a in text.split(",") if a.strip()]


# ============================================================
# MARKDOWN CLEANER (USED EVERYWHERE)
# ============================================================

def clean_markdown(text: str) -> str:
    if not text:
        return ""

    # Remove markdown headings
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold / italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # Remove bullets and numbering
    text = re.sub(r"^\s*-\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

    return text.strip()


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
    return clean_markdown(output), roles


# ============================================================
# ROLE-LOCKED AGENT EXECUTION
# ============================================================

def run_agent(task, agent, history):
    system = f"""
You are acting STRICTLY as a {agent}.

ROLE CONSTRAINTS:
- Speak ONLY from the perspective of a {agent}
- Do NOT describe the full end-to-end solution
- Do NOT summarize or conclude the overall workflow
- Do NOT repeat other roles
- Focus ONLY on decisions, tools, risks, and recommendations
  that belong to this role

ASSUME:
- Other agents exist
- A Summary Agent will synthesize

STYLE:
- Professional consulting tone
- Short focused paragraphs
- No markdown, no bullets, no emojis
"""

    recent_context = "\n".join(history.split("\n")[-12:])

    user = f"""
TASK:
{task}

RECENT CONTEXT (DO NOT REPEAT):
{recent_context}

Now contribute ONLY from the {agent} role.
"""
    return clean_markdown(llm(system, user))


def summarize(task, history):
    return clean_markdown(
        llm(
            "You are the Summary Agent. Produce an executive summary with insights, trade-offs, and next steps.",
            f"TASK:\n{task}\n\nFULL CONTEXT:\n{history}"
        )
    )


# ============================================================
# CHAT RENDERER (CLEAN FRONTEND)
# ============================================================

def alignment(agent, index):
    if agent.lower().startswith("summary"):
        return "center"
    return "left" if index % 2 == 0 else "right"


def render_chat(agent_log):
    html = """
    <style>
    .bubble, .bubble * { color:#000 !important; }
    .bubble {
      max-width:70%;
      padding:14px;
      margin:10px;
      border-radius:14px;
      white-space:pre-wrap;
      box-shadow:0 1px 4px rgba(0,0,0,0.15);
    }
    .left { background:#E5E7EB; margin-right:auto; }
    .right { background:#FEF3C7; margin-left:auto; }
    .center {
      background:#EDE9FE;
      margin:20px auto;
      text-align:center;
      font-weight:600;
    }
    .agent {
      font-size:12px;
      font-weight:700;
      margin-bottom:8px;
      color:#F97316 !important;
      text-transform:uppercase;
      letter-spacing:0.04em;
    }
    .typing { font-style:italic; color:#374151 !important; }
    </style>
    <div>
    """

    for idx, (agent, text) in enumerate(agent_log.items()):
        side = alignment(agent, idx)
        html += f"""
        <div class="bubble {side}">
          <div class="agent">{agent}</div>
          {clean_markdown(text)}
        </div>
        """

    html += "</div>"
    return html


# ============================================================
# PDF FORMATTER (PROFESSIONAL)
# ============================================================

def format_agent_for_pdf(agent, text):
    return [
        ("Role Focus", f"The {agent} provided insights specific to their functional responsibility."),
        ("Key Insights", clean_markdown(text)),
        ("Recommendations", f"Recommendations from the {agent} should be evaluated and synthesized in the final decision.")
    ]


def export_pdf(task, log, summary):
    path = f"/tmp/orbita_{uuid.uuid4().hex}.pdf"
    styles = getSampleStyleSheet()

    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    header = ParagraphStyle("header", parent=styles["Heading2"])
    sub = ParagraphStyle("sub", parent=styles["Heading4"])

    doc = SimpleDocTemplate(path, pagesize=A4)
    story = []

    if os.path.exists(LOGO_PATH):
        story.append(RLImage(LOGO_PATH, width=3 * inch, height=1 * inch))
    story.append(Spacer(1, 14))

    story.append(Paragraph(TAGLINE, styles["Heading2"]))
    story.append(Paragraph(f"{AUTHOR} • {AUTHOR_LINKEDIN}", body))
    story.append(Spacer(1, 18))

    story.append(Paragraph("Task Overview", header))
    story.append(Paragraph(clean_markdown(task), body))
    story.append(PageBreak())

    for agent, text in log.items():
        if agent.lower().startswith("summary"):
            continue

        story.append(Paragraph(f"Role: {agent}", header))
        story.append(Spacer(1, 6))

        for title, content in format_agent_for_pdf(agent, text):
            story.append(Paragraph(title, sub))
            story.append(Paragraph(content, body))
            story.append(Spacer(1, 10))

        story.append(Spacer(1, 18))

    story.append(PageBreak())
    story.append(Paragraph("Executive Summary", header))
    story.append(Spacer(1, 10))

    for line in summary.split("\n"):
        story.append(Paragraph(clean_markdown(line), body))

    doc.build(story)
    return path


# ============================================================
# CORE WORKFLOW
# ============================================================

def run_all(task, agent_text):
    planner_out, suggested_agents = planner(task)
    agents = clean_list(agent_text) if agent_text.strip() else suggested_agents

    log = {"Planner": planner_out}
    history = planner_out

    for agent in agents:
        log[agent] = "Agent thinking..."
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

    run_btn = gr.Button("Run ORBITA")

    chat_view = gr.HTML()
    summary_box = gr.Textbox(label="Executive Summary", lines=8)
    state_log = gr.State({})

    pdf_btn = gr.Button("Export Professional PDF")
    pdf_file = gr.File()

    def suggest_agents(task):
        _, roles = planner(task)
        return ", ".join(roles)

    suggest_btn.click(suggest_agents, task_box, agents_box)

    run_btn.click(
        run_all,
        inputs=[task_box, agents_box],
        outputs=[chat_view, state_log, summary_box]
    )

    pdf_btn.click(
        lambda t, l, s: export_pdf(t, l, s),
        inputs=[task_box, state_log, summary_box],
        outputs=pdf_file
    )

app.queue()
app.launch()
