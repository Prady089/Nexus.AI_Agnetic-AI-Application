import os, re
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import json
import asyncio

load_dotenv()

app = FastAPI()

# Configuration
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace")
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Mount entire workspace (sessions live as subdirs)
app.mount("/workspace", StaticFiles(directory=WORKSPACE_DIR), name="workspace")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ─── LLM ─────────────────────────────────────────────────────────────────────
def llm(system, user, temperature=0.7):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-3.5-flash-lite",
            system_instruction=system + "\n\nImportant: Do not recite copyrighted material or training data verbatim. Be creative and synthesize new logic based on requirements."
        )
        response = model.generate_content(
            user,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        if hasattr(response, 'candidates') and response.candidates:
            if response.candidates[0].finish_reason == 4:
                return "AGENT ERROR: Recitation block. Try re-phrasing your request."
        return response.text.strip()
    except Exception as e:
        return f"LLM_ERROR: {str(e)}"

# ─── FILE HELPERS ─────────────────────────────────────────────────────────────
def extract_files(text, session_dir):
    """Parse ### FILE: blocks and write them into session_dir."""
    pattern = r"### FILE:\s*([a-zA-Z0-9_\-\.]+)\n+```[a-z]*\n([\s\S]*?)```"
    matches = re.finditer(pattern, text)
    created = []
    for match in matches:
        filename = match.group(1).strip()
        content = match.group(2)
        with open(os.path.join(session_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
        created.append(filename)
    return created

def read_session_files(session_dir):
    """Return all file contents in a session as a context string."""
    context = ""
    try:
        files = [f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f))]
        for f in files:
            try:
                with open(os.path.join(session_dir, f), "r", encoding="utf-8") as c:
                    context += f"\n--- FILE: {f} ---\n{c.read()[:3000]}\n"
            except:
                pass
    except:
        pass
    return context

def strip_code_blocks(text):
    """Get a clean human-readable summary (non-code parts only)."""
    clean = re.sub(r'```[\s\S]*?```', '[code generated]', text)
    clean = re.sub(r'### FILE:.*', '', clean)
    clean = ' '.join(clean.split())
    return clean[:600]

def get_session_label(sid):
    """Convert session_20260818_112600 -> Aug 18, 11:26"""
    try:
        dt = datetime.strptime(sid, "session_%Y%m%d_%H%M%S")
        return dt.strftime("%b %d, %H:%M:%S")
    except:
        return sid

# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/sessions")
async def list_sessions():
    """List all session folders sorted newest first."""
    sessions = []
    try:
        entries = sorted(
            [e for e in os.listdir(WORKSPACE_DIR)
             if os.path.isdir(os.path.join(WORKSPACE_DIR, e)) and e.startswith("session_")],
            reverse=True
        )
        for sid in entries:
            path = os.path.join(WORKSPACE_DIR, sid)
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            sessions.append({
                "id": sid,
                "label": get_session_label(sid),
                "files": files,
                "file_count": len(files)
            })
    except:
        pass
    return JSONResponse(sessions)

# ─── CHAT ─────────────────────────────────────────────────────────────────────
@app.get("/chat")
async def chat_with_agent(message: str, session_id: str = ""):
    async def event_generator():
        try:
            context = ""
            if session_id:
                session_dir = os.path.join(WORKSPACE_DIR, session_id)
                context = read_session_files(session_dir)

            sys_prompt = """You are the Nexus.AI Project Manager — a sharp, conversational AI embedded in an agentic software factory.
You oversee a team: Planner, Business Analyst, Architect, Developer, QA.
You have full visibility into the current session's files.
Be specific — reference actual filenames, code details, feature decisions.
Keep responses focused and conversational (2-5 sentences). Never say 'I cannot'."""

            user_msg = (f"Session files:\n{context}\n\nUser: {message}" if context
                        else f"No project built yet.\nUser: {message}\n\nTell them to describe a requirement to start building.")

            reply = llm(sys_prompt, user_msg, temperature=0.8)
            yield f"data: {json.dumps({'event': 'chat', 'agent': 'PROJECT MANAGER', 'text': reply})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'chat', 'agent': 'SYSTEM', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ─── INTENT ───────────────────────────────────────────────────────────────────
@app.get("/intent")
async def detect_intent(message: str):
    sys_prompt = """Intent classifier. Reply with exactly one word: 'build' or 'chat'.
'build' = user wants to CREATE, MAKE, BUILD, ADD, CHANGE, FIX, UPDATE, REFINE, GENERATE software.
'chat'  = user is asking a QUESTION, checking STATUS, saying hi, or having a conversation."""
    result = llm(sys_prompt, message, temperature=0.1).strip().lower()
    return {"intent": "build" if "build" in result else "chat"}

# ─── BUILD / RUN ──────────────────────────────────────────────────────────────
@app.get("/run")
async def run_factory(task: str, mode: str = "initial", session_id: str = ""):
    async def event_generator():
        try:
            # ── SESSION SETUP ─────────────────────────────────────────
            if mode == "initial" or not session_id:
                sid = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            else:
                sid = session_id

            session_dir = os.path.join(WORKSPACE_DIR, sid)
            os.makedirs(session_dir, exist_ok=True)

            # Tell frontend which session this is
            yield f"data: {json.dumps({'event': 'session', 'session_id': sid, 'label': get_session_label(sid)})}\n\n"

            is_refine = (mode == "refine" and bool(session_id))
            existing_context = read_session_files(session_dir) if is_refine else ""

            # ── PLAN ──────────────────────────────────────────────────
            yield f"data: {json.dumps({'event': 'node', 'node': 'node-planner'})}\n\n"

            plan_sys = """You are a Senior Software Architect and Project Planner.
Given a task, assign exactly 4 specialist roles (comma-separated).
Each role should be specific: e.g. 'Business Analyst', 'System Architect', 'Full-Stack Developer', 'QA Engineer'.
Return only the 4 roles as a comma-separated list. No extra text."""
            plan_user = f"TASK: {task}" + (f"\nEXISTING CODE:\n{existing_context[:800]}" if is_refine else "")

            roles_raw = llm(plan_sys, plan_user)
            roles = [r.strip() for r in roles_raw.split(",")][:4]
            while len(roles) < 4:
                roles.append("Full-Stack Developer")

            team_msg = f"Assembled team: {', '.join(roles)}. Session: {get_session_label(sid)}. Briefing agents and distributing work packages."
            yield f"data: {json.dumps({'event': 'log', 'agent': 'PLANNER', 'text': team_msg})}\n\n"
            await asyncio.sleep(0.3)

            history = f"TASK: {task}\n\n"
            if is_refine:
                history += f"CURRENT PROJECT STATE:\n{existing_context}\n\n"

            # ── AGENT PROMPTS ─────────────────────────────────────────
            nodes = ["node-ba", "node-arch", "node-dev", "node-qa"]

            agent_system_prompts = {
                0: lambda role: f"""You are a {role} working on a real software project.
Analyze the task deeply and produce a detailed Business Requirements Document (BRD).
You MUST output your BRD using EXACTLY this format so it gets saved as a file:

### FILE: brd.md
```markdown
# Business Requirements Document
[full BRD content]
```

The BRD must cover: project overview, numbered functional requirements, user stories (As a [user] I want [X] so that [Y]), key features with acceptance criteria, technical constraints, and assumptions. Be thorough — this drives the entire build.""",

                1: lambda role: f"""You are a {role} designing a software system.
Based on the BRD in the project history, define the full technical architecture.
You MUST output your architecture document using EXACTLY this format so it gets saved:

### FILE: architecture.md
```markdown
# Technical Architecture
[full architecture document]
```

Cover: technology stack with reasoning, complete file structure (list every file), component responsibilities, data models and state management, key algorithms, and how components interact.""",

                2: lambda role: f"""You are a {role} implementing production-ready code.
Write complete, working code using plain HTML/CSS/JS only (no external frameworks).
For EACH file, output it using EXACTLY this format:

### FILE: filename.ext
```language
[complete file content]
```

Write ALL files needed (index.html, style.css, script.js at minimum). Each file must be fully complete with no placeholders. Use modern CSS (gradients, shadows, transitions) for a polished UI.""",

                3: lambda role: f"""You are a {role} reviewing the delivered software.
Produce a detailed QA Report. You MUST output it using EXACTLY this format so it gets saved:

### FILE: qa_report.md
```markdown
# QA Report
[full QA report]
```

Include: executive summary with quality rating /10, feature verification table (PASS/FAIL/PARTIAL), bugs found with severity (Critical/Major/Minor), unhandled edge cases, and specific improvement recommendations.""",
            }

            # ── RUN AGENTS ────────────────────────────────────────────
            for i, agent in enumerate(roles):
                node_id = nodes[i] if i < len(nodes) else "node-dev"
                yield f"data: {json.dumps({'event': 'node', 'node': node_id})}\n\n"

                sys_prompt = agent_system_prompts[i](agent)
                user_prompt = f"Project history:\n{history[-4000:]}\n\n{'USER CHANGE REQUEST' if is_refine else 'TASK'}: {task}\n\nDeliver your work now."

                output = llm(sys_prompt, user_prompt)
                files_created = extract_files(output, session_dir)

                summary = strip_code_blocks(output)
                if not summary or len(summary) < 20:
                    summary = f"{agent} phase completed."

                history += f"\n\n{agent} Output:\n{output}"

                yield f"data: {json.dumps({'event': 'log', 'agent': agent.upper(), 'text': summary, 'files': files_created, 'session_id': sid})}\n\n"
                await asyncio.sleep(0.4)

            count = len([f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f))])
            yield f"data: {json.dumps({'event': 'log', 'agent': 'RELEASE MANAGER', 'text': f'Build complete. {count} artifacts saved to session {get_session_label(sid)}. Switch sessions from the History tab.'})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'log', 'agent': 'SYSTEM ERROR', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
