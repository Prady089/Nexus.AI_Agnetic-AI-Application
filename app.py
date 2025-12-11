import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
from typing import List, Tuple

import gradio as gr
from agent import Agent, Runner


# ============================================================
# 0. LOAD API KEY FROM HUGGING FACE SECRETS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Add it in HuggingFace → Settings → Variables.")

# ============================================================
# 1. PLANNER AGENT – suggests personas + workflow
# ============================================================

planner_instructions = """
You are an expert orchestrator AI that designs teams of specialist agents.

Given a TASK, you must:
1) Identify what type of problem this is (e.g., software project, trip planning, career planning, etc.).
2) Propose a set of 3–7 personas (roles) that should work together to solve it.
3) For each persona, explain briefly WHY they are needed.
4) Propose a LINEAR workflow order.
5) Ensure roles in the workflow EXACTLY match persona names.

Your output MUST follow:

Personas:
1. <Role 1> - <short justification>
2. <Role 2> - <short justification>
...

Workflow:
Describe the flow in 2–4 sentences.

Linear_Workflow_Roles: <Role 1>, <Role 2>, <Role 3>, ...
"""

planner_agent = Agent(
    name="Planner",
    instructions=planner_instructions,
    model="gpt-4o-mini"
)


async def planner_suggest(task: str):
    """Return (full text, roles list)."""
    prompt = f"TASK:\n{task}\n\nFollow the required structure."
    result = await Runner.run(planner_agent, prompt)

    full_text = result.final_output

    # Extract roles from Linear_Workflow_Roles line
    roles = []
    for line in full_text.splitlines():
        if line.startswith("Linear_Workflow_Roles:"):
            _, _, rest = line.partition(":")
            roles = [r.strip() for r in rest.split(",") if r.strip()]
            break

    return full_text, roles


# ============================================================
# 2. ROLE AGENT FACTORY
# ============================================================

def make_role_agent(role: str) -> Agent:
    instructions = f"""
You are acting in the role: {role}.

You are part of a multi-expert task force.
Your output must advance the task logically and professionally.

Follow this:
- Read the TASK.
- Read TEAM HISTORY.
- Contribute ONLY from the perspective of {role}.
- Add new, meaningful progress.
- Be structured, concise, and useful.

Format with headings and bullet points.
"""
    return Agent(
        name=role,
        instructions=instructions,
        model="gpt-4o-mini"
    )


async def run_role_step(task: str, role: str, history: str) -> str:
    agent = make_role_agent(role)
    prompt = f"""
TASK:
{task}

TEAM HISTORY:
{history or "(empty)"}

Now act as the {role} and advance the work.
"""
    result = await Runner.run(agent, prompt)
    return result.final_output


# ============================================================
# 3. EVALUATOR AGENT — final summary
# ============================================================

evaluator_agent = Agent(
    name="Evaluator",
    instructions="""
You summarize the complete multi-agent workflow.

Your summary MUST include:
- High-level overview
- Key decisions made by each role
- Final actionable output
- Risks / Next Steps (if applicable)
    """,
    model="gpt-4o-mini"
)


async def evaluate_workflow(task, history):
    prompt = f"""
TASK:
{task}

FULL WORKFLOW HISTORY:
{history}

Please produce a structured final summary.
"""
    result = await Runner.run(evaluator_agent, prompt)
    return result.final_output


# ============================================================
# 4. FULLY AUTOMATED WORKFLOW EXECUTION
# ============================================================

async def run_full_automation(task: str):
    """Runs planner → role1 → role2 → ... → evaluator automatically."""
    if not task.strip():
        return "⚠️ Please enter a task.", ""

    # 1) Get personas + workflow
    plan_text, roles = await planner_suggest(task)

    if not roles:
        return plan_text + "\n\n❌ No workflow roles detected.", ""

    history = f"=== PLANNER RECOMMENDATION ===\n{plan_text}\n"

    # 2) Run each role AUTOMATICALLY
    for step, role in enumerate(roles, start=1):
        step_output = await run_role_step(task, role, history)
        history += f"\n\n=== STEP {step}: {role} ===\n{step_output}\n"

    # 3) Final evaluator summary
    summary = await evaluate_workflow(task, history)

    return history, summary


# ============================================================
# 5. GRADIO UI — Hugging Face Ready
# ============================================================

def run_full(task):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(run_full_automation(task))


with gr.Blocks() as app:
    gr.Markdown("# 🧠 Adaptive Multi-Agent Orchestrator (AMAO)")
    gr.Markdown(
        "Provide ANY task (software project, trip plan, study plan, business strategy). "
        "AMAO automatically selects personas, runs the workflow end-to-end, and produces a final evaluator summary."
    )

    task_box = gr.Textbox(
        label="Enter Your Task",
        placeholder="Example: Plan a 6-day Singapore trip under $1500.\nOr: Build MVP for Guardian safety app.",
        lines=4
    )

    run_btn = gr.Button("🚀 Run Automated Multi-Agent Workflow")

    workflow_log = gr.Textbox(label="Full Workflow Log", lines=22)
    final_summary = gr.Textbox(label="Evaluator Summary", lines=16)

    run_btn.click(
        fn=run_full,
        inputs=task_box,
        outputs=[workflow_log, final_summary]
    )

app.launch()
