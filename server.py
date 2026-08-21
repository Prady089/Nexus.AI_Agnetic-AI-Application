import os, re
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
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

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── LLM ─────────────────────────────────────────────────────────────────────
def llm(system, user, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM_ERROR: {str(e)}"

# ─── FILE HELPERS ─────────────────────────────────────────────────────────────
def walk_session_files(session_dir):
    """Return all file paths in a session dir (relative, posix-style), recursively."""
    paths = []
    for root, _, files in os.walk(session_dir):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), session_dir).replace(os.sep, "/")
            paths.append(rel)
    return sorted(paths)

def extract_files(text, session_dir):
    """Parse ### FILE: blocks (including subdirectory paths) and write them into session_dir."""
    pattern = r"### FILE:\s*([a-zA-Z0-9_\-./\\]+)\n+```[a-z]*\n([\s\S]*?)```"
    matches = re.finditer(pattern, text)
    created = []
    for match in matches:
        rel_path = match.group(1).strip().replace("\\", "/").lstrip("/")
        parts = [p for p in rel_path.split("/") if p not in ("", ".", "..")]
        if not parts:
            continue
        rel_path = "/".join(parts)
        content = match.group(2)
        dest = os.path.join(session_dir, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        created.append(rel_path)
    return created

def read_session_files(session_dir):
    """Return all file contents in a session as a context string."""
    context = ""
    try:
        for f in walk_session_files(session_dir):
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
            files = walk_session_files(path)
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
            session_dir = None
            if session_id:
                session_dir = os.path.join(WORKSPACE_DIR, session_id)
                context = read_session_files(session_dir)

            sys_prompt = """You are the Nexus.AI Project Manager — a sharp, conversational AI embedded in an agentic software factory.
You oversee a team: Business Analyst, Architect, Developer, Quality Analyst.
You have full visibility into the current session's files.
Be specific — reference actual filenames, code details, feature decisions.
Keep responses focused and conversational (2-5 sentences). Never say 'I cannot'."""

            user_msg = (f"Session files:\n{context}\n\nUser: {message}" if context
                        else f"No project built yet.\nUser: {message}\n\nTell them to describe a requirement to start building.")

            reply = llm(sys_prompt, user_msg, temperature=0.8)
            yield f"data: {json.dumps({'event': 'chat', 'agent': 'PROJECT MANAGER', 'text': reply})}\n\n"

            # If there's an active project, have the Developer actually apply
            # any code change this conversation implies - a chat reply alone
            # never touches files, so without this "amendments" are just talk.
            if session_dir and context:
                dev_sys = """You are a Developer amending an EXISTING project based on a conversation with the Project Manager.
You are given the CURRENT files and the latest exchange. If it requires a code change, output ONLY the files that need to change, using EXACTLY this format:

### FILE: filename.ext
```language
[complete updated file content]
```

You MUST preserve the exact existing file paths and names — do not rename, move, or restructure the project. If nothing in the exchange requires a code change (e.g. it's a question or general conversation), respond with just: NO_CHANGE_NEEDED"""
                dev_user = f"Existing files:\n{context}\n\nProject Manager's note: {reply}\n\nUser message: {message}\n\nApply any necessary code change now."

                yield f"data: {json.dumps({'event': 'node', 'node': 'node-dev'})}\n\n"
                dev_output = llm(dev_sys, dev_user, temperature=0.4)
                files_changed = extract_files(dev_output, session_dir)
                if files_changed:
                    summary = strip_code_blocks(dev_output)
                    if not summary or len(summary) < 20:
                        summary = "Applied the requested change."
                    yield f"data: {json.dumps({'event': 'log', 'agent': 'DEVELOPER', 'text': summary, 'files': files_changed, 'session_id': session_id})}\n\n"

            yield f"data: {json.dumps({'event': 'complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'chat', 'agent': 'SYSTEM', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ─── INTENT ───────────────────────────────────────────────────────────────────
@app.get("/intent")
async def detect_intent(message: str):
    sys_prompt = """Intent classifier. Reply with exactly one word: 'build' or 'chat'.
'build' = user wants to CREATE, MAKE, BUILD, ADD, CHANGE, FIX, UPDATE, REFINE, GENERATE software,
          OR is reporting that the built app is broken/not working/has a bug (this requires a code fix, so it is 'build').
'chat'  = user is asking a QUESTION about the project, saying hi, or having a conversation that does not require any code change."""
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
            # Roles are fixed to match the sidebar identities exactly, not
            # LLM-invented each run - otherwise the displayed agent name has
            # no relation to which fixed-prompt phase (BRD/architecture/code/QA)
            # actually produced the content.
            yield f"data: {json.dumps({'event': 'node', 'node': 'node-planner'})}\n\n"

            if is_refine:
                roles = ["Developer", "Quality Analyst"]
                phase_indices = [2, 3]
                nodes = ["node-dev", "node-qa"]
                team_msg = f"Focused fix team: {', '.join(roles)}. Session: {get_session_label(sid)}. Applying the requested change directly to the existing build."
            else:
                roles = ["Business Analyst", "Architect", "Developer", "Quality Analyst"]
                phase_indices = [0, 1, 2, 3]
                nodes = ["node-ba", "node-arch", "node-dev", "node-qa"]
                team_msg = f"Assembled team: {', '.join(roles)}. Session: {get_session_label(sid)}. Briefing agents and distributing work packages."

            yield f"data: {json.dumps({'event': 'log', 'agent': 'PROJECT MANAGER', 'text': team_msg})}\n\n"
            await asyncio.sleep(0.3)

            history = f"TASK: {task}\n\n"
            if is_refine:
                history += f"CURRENT PROJECT STATE:\n{existing_context}\n\n"

            # ── AGENT PROMPTS ─────────────────────────────────────────
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

Write ALL files needed (index.html, style.css, script.js at minimum). Each file must be fully complete with no placeholders. Use modern CSS (gradients, shadows, transitions) for a polished UI.

LAYOUT RULES (critical - do not skip):
- Include <meta name="viewport" content="width=device-width, initial-scale=1.0"> in the HTML head.
- This is a web app viewed in a full desktop browser. Do NOT constrain the whole page to a narrow mobile-width column (e.g. 300-400px). Dashboards, calculators, and multi-field tools must use a wide, centered content area (roughly 700-1100px) or the full viewport width with responsive padding - only login/auth cards should be a narrow centered card.
- Use CSS Grid or Flexbox to lay related fields/results side-by-side on desktop where it makes sense, collapsing to a single column only below 768px via a media query.
- Format currency and large numbers with thousands separators (e.g. $928,405.00, not $928405.00) using toLocaleString() or equivalent.
- If you draw any chart, icon, or graphic with Canvas/SVG, use precise arc/path math (correct radius, center, and angle calculations for circles/donuts) - never approximate with irregular polygon points that render as a distorted blob.""",

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
            for agent, phase_idx, node_id in zip(roles, phase_indices, nodes):
                yield f"data: {json.dumps({'event': 'node', 'node': node_id})}\n\n"

                sys_prompt = agent_system_prompts[phase_idx](agent)
                if is_refine and phase_idx == 2:
                    sys_prompt += "\n\nIMPORTANT: This is an amendment to an EXISTING project. You MUST preserve the exact existing file paths and names shown in the project history below - do not invent a new folder structure, do not rename or move files. Only output the files that actually need to change to fulfill the request."
                user_prompt = f"Project history:\n{history[-4000:]}\n\n{'USER CHANGE REQUEST' if is_refine else 'TASK'}: {task}\n\nDeliver your work now."

                output = llm(sys_prompt, user_prompt)
                files_created = extract_files(output, session_dir)

                summary = strip_code_blocks(output)
                if not summary or len(summary) < 20:
                    summary = f"{agent} phase completed."

                history += f"\n\n{agent} Output:\n{output}"

                yield f"data: {json.dumps({'event': 'log', 'agent': agent.upper(), 'text': summary, 'files': files_created, 'session_id': sid})}\n\n"
                await asyncio.sleep(0.4)

            count = len(walk_session_files(session_dir))
            yield f"data: {json.dumps({'event': 'log', 'agent': 'RELEASE MANAGER', 'text': f'Build complete. {count} artifacts saved to session {get_session_label(sid)}. Switch sessions from the History tab.'})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'log', 'agent': 'SYSTEM ERROR', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
