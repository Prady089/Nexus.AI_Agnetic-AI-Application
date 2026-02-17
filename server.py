import os, re, time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import json
import asyncio

app = FastAPI()

# Configuration
WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace")
if not os.path.exists(WORKSPACE_DIR):
    os.makedirs(WORKSPACE_DIR)

# Mount Workspace for Live Preview
app.mount("/workspace", StaticFiles(directory=WORKSPACE_DIR), name="workspace")

# Hardcoded Key from user
genai.configure(api_key="AIzaSyCtiYJt8vyQvi7qXh4Iqv0F-3iZEi2gfrA")

def llm(system, user):
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system + "\n\nImportant: Do not recite copyrighted material or training data verbatim. Be creative and synthesize new logic based on requirements."
        )
        response = model.generate_content(
            user,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
            )
        )
        
        # Check for recitation block
        if hasattr(response, 'candidates') and response.candidates:
            if response.candidates[0].finish_reason == 4:
                return "AGENT ERROR: Recitation block (Finish Reason 4). Try re-phrasing your request slightly."
        
        return response.text.strip()
    except Exception as e:
        return f"LLM_ERROR: {str(e)}"

def extract_files(text):
    pattern = r"### FILE:\s*([a-zA-Z0-9_\-\.]+)\n+```[a-z]*\n([\s\S]*?)```"
    matches = re.finditer(pattern, text)
    created = []
    for match in matches:
        filename = match.group(1).strip()
        content = match.group(2)
        with open(os.path.join(WORKSPACE_DIR, filename), "w", encoding="utf-8") as f:
            f.write(content)
        created.append(filename)
    return created

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/run")
async def run_factory(task: str, mode: str = "initial"):
    async def event_generator():
        try:
            is_refine = mode == "refine"
            
            # 1. READ EXISTING FILES FOR CONTEXT (IF REFINING)
            existing_context = ""
            if is_refine:
                files = [f for f in os.listdir(WORKSPACE_DIR) if os.path.isfile(os.path.join(WORKSPACE_DIR, f))]
                for f in files:
                    with open(os.path.join(WORKSPACE_DIR, f), "r", encoding="utf-8") as content:
                        existing_context += f"\n--- FILE: {f} ---\n{content.read()}\n"

            # 2. PLAN
            yield f"data: {json.dumps({'event': 'node', 'node': 'node-planner'})}\n\n"
            
            plan_sys = "You are a Software Architect. Plan 4 roles for the task. Return as CSV only."
            plan_user = f"TASK: {task}\nEXISTING CODE:\n{existing_context}" if is_refine else task
            
            roles_raw = llm(plan_sys, plan_user)
            roles = [r.strip() for r in roles_raw.split(",")]
            
            history = f"TASK: {task}\n\n"
            if is_refine: history += f"CURRENT PROJECT STATE:\n{existing_context}\n\n"
            
            # 3. RUN AGENTS
            nodes = ["node-ba", "node-arch", "node-dev", "node-qa"]
            for i, agent in enumerate(roles):
                node_id = nodes[i] if i < len(nodes) else "node-dev"
                yield f"data: {json.dumps({'event': 'node', 'node': node_id})}\n\n"
                
                system_prompt = f"You are a {agent} in a Refinement Loop. Your goal is to MODIFY the existing code based on user feedback."
                if "Developer" in agent:
                    system_prompt += " Use plain HTML/JS only. Return full updated files with ### FILE: filename.ext."
                
                output = llm(system_prompt, f"History: {history[-3000:]}\n\nUSER FEEDBACK: {task}\n\nApply the change.")
                files_created = extract_files(output)
                history += f"\n\n{agent} Output:\n{output}"
                
                yield f"data: {json.dumps({'event': 'log', 'agent': agent.upper(), 'text': 'Refining artifacts...', 'files': files_created})}\n\n"
                await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'event': 'log', 'agent': 'RELEASE MANAGER', 'text': 'Refinement complete. Workspace updated.'})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'log', 'agent': 'SYSTEM ERROR', 'text': str(e)})}\n\n"
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
