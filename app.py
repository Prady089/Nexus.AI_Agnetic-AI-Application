import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
import gradio as gr
from typing import List, Tuple

from openai import OpenAI


# ============================================================
# API KEY
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it in HuggingFace → Settings → Variables and Secrets."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# GENERIC LLM CALL
# ============================================================

def run_llm(system_prompt: str, user_prompt: str, model="gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3
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
2. <Role> - <Why needed>

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
- Only contribute from this role’s perspective
- Read TASK and TEAM HISTORY
- Do NOT repeat earlier content
- Move the work forward
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
- Each agent’s contribution
- Final synthesized result
- Clear next steps or recommendations
"""


def run_summary(task: str, history: str) -> str:
    return run_llm(
        SUMMARY_PROMPT,
        f"TASK:\n{task}\n\nFULL WORKFLOW LOG:\n{history}"
    )


# ============================================================
# AUTOMATION ENGINE
# ============================================================

async def run_automation(task: str, selected_agents: List[str]):
    if not task.strip():
        return "⚠️ Please enter a task.", ""

    plan_text, suggested_agents = await planner_suggest(task)

    history = "=== PLANNER OUTPUT ===\n" + plan_text + "\n"

    # Use planner order but filter by user selection
    execution_agents = [a for a in suggested_agents if a in selected_agents]

    if not execution_agents:
        return history + "\n❌ No agents selected.", ""

    for idx, agent in enumerate(execution_agents, start=1):
        output = run_role(task, agent, history)
        history += f"\n\n=== STEP {idx}: {agent} ===\n{output}\n"

    # Summary agent always runs last
    summary = run_summary(task, history)

    return history, summary


def run_sync(task, agents):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_automation(task, agents))


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown("# 🤖 Automatic Multi-Agent Workflow")
    gr.Markdown(
        "Agents are planned, selected, and executed automatically. "
        "A Summary Agent always runs at the end."
    )

    task_box = gr.Textbox(
        label="Enter Task",
        lines=4,
        placeholder="Example: Build an AI-powered resume analyzer."
    )

    agent_selector = gr.CheckboxGroup(
        label="Select Agents to Run (Summary Agent runs automatically)",
        choices=[],
        interactive=True
    )

    plan_btn = gr.Button("🧠 Generate Agent Plan")
    run_btn = gr.Button("🚀 Run Automatic Workflow")

    workflow_log = gr.Textbox(label="Full Workflow Log", lines=22)
    final_summary = gr.Textbox(label="Final Summary (Auto)", lines=14)

    # Planner step populates agent selector
    async def populate_agents(task):
        plan, agents = await planner_suggest(task)
        return gr.CheckboxGroup(choices=agents, value=agents)

    plan_btn.click(
        populate_agents,
        inputs=task_box,
        outputs=agent_selector
    )

    run_btn.click(
        run_sync,
        inputs=[task_box, agent_selector],
        outputs=[workflow_log, final_summary]
    )

app.launch()
