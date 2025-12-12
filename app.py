import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
from typing import List, Tuple

import gradio as gr

# Import the correct agents library (bundled inside openai>=1.6.0)
from agents import Agent, Runner


# ============================================================
# LOAD API KEY
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it in HuggingFace → Settings → Variables and Secrets."
    )


# ============================================================
# PLANNER AGENT
# ============================================================

planner_instructions = """
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

planner_agent = Agent(
    name="Planner",
    instructions=planner_instructions,
    model="gpt-4o-mini"
)


async def planner_suggest(task: str):
    prompt = f"TASK:\n{task}\n\nFollow the required output structure."
    result = await Runner.run(planner_agent, prompt)

    output = result.final_output
    roles = []

    for line in output.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            _, _, rest = line.partition(":")
            roles = [r.strip() for r in rest.split(",") if r.strip()]
            break

    return output, roles


# ============================================================
# ROLE AGENT FACTORY
# ============================================================

def make_role_agent(role: str):
    instructions = f"""
You are acting in the role: {role}.
You contribute ONLY from the perspective of this role.

Your responsibilities:
- Read the TASK.
- Read TEAM HISTORY.
- Continue the work logically.
- Do not repeat previous content.
- Provide valuable forward progress.
- Use headings & bullet points.
"""
    return Agent(name=role, instructions=instructions, model="gpt-4o-mini")


async def run_role_step(task: str, role: str, history: str):
    agent = make_role_agent(role)
    prompt = f"""
TASK:
{task}

TEAM HISTORY:
{history or "(none)"}

Now act as the {role} and advance the work.
"""
    result = await Runner.run(agent, prompt)
    return result.final_output


# ============================================================
# EVALUATOR AGENT
# ============================================================

evaluator_agent = Agent(
    name="Evaluator",
    instructions="""
You summarize the complete multi-agent workflow.

Your summary MUST include:
- Overview of the task
- Key contributions from each agent
- Final synthesized output
- Next steps OR recommendations
""",
    model="gpt-4o-mini"
)


async def evaluate_workflow(task: str, history: str):
    prompt = f"""
TASK:
{task}

FULL TEAM HISTORY:
{history}

Please produce a CLEAR, STRUCTURED, FINAL SUMMARY.
"""
    result = await Runner.run(evaluator_agent, prompt)
    return result.final_output


# ============================================================
# MASTER AUTOMATION ENGINE
# ============================================================

async def run_full_automation(task: str):
    if not task.strip():
        return "⚠️ Please enter a task.", ""

    # Planner
    plan_text, roles = await planner_suggest(task)
    if not roles:
        return plan_text + "\n\n❌ No roles detected.", ""

    history = "=== PLANNER RECOMMENDATION ===\n" + plan_text + "\n"

    # Auto-run each role
    for i, role in enumerate(roles, start=1):
        role_output = await run_role_step(task, role, history)
        history += f"\n\n=== STEP {i}: {role} ===\n{role_output}\n"

    # Final evaluation summary
    summary = await evaluate_workflow(task, history)

    return history, summary


def run_sync(task):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_full_automation(task))


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as app:
    gr.Markdown("# 🧠 Adaptive Multi-Agent Orchestrator (AMAO)")
    gr.Markdown("Enter ANY task. AMAO will auto-select roles, run them, and generate a final evaluated summary.")

    task_box = gr.Textbox(
        label="Enter Task",
        placeholder="Example: Build MVP of a fitness tracking app.\nOr: Plan a 5-day Tokyo trip.",
        lines=4
    )

    run_btn = gr.Button("🚀 Run Automated Workflow")

    workflow_log = gr.Textbox(label="Full Workflow Log", lines=22)
    final_summary = gr.Textbox(label="Evaluator Summary", lines=14)

    run_btn.click(run_sync, inputs=task_box, outputs=[workflow_log, final_summary])

app.launch()
