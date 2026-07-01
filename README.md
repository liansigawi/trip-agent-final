# Trip Planning AI Agent (Python, Vercel native functions)

Autonomous trip planner. One natural-language request → profiled traveller → ReAct planning loop
over travel tools → self-critique → costed day-by-day itinerary.

## Pipeline (module names match the diagram and the `steps[]` trace)
`Preference Profiler` → `ReAct Planner` (⇄ `Reflection Layer`, max 2 re-plan cycles) → `Output Formatter`

Tools: `maps_tool`, `search_tool`, `reviews_tool`, `weather_tool` (live), `flights_tool` & `booking_tool`
(**fictive** — never make a real reservation/purchase), `calendar_tool`. Tier-2 `booking_confirm_tool` /
`flight_book_tool` are gated and never auto-fire.

## Layout
```
api/team_info.py            GET  /api/team_info          (edit your details here)
api/agent_info.py           GET  /api/agent_info
api/model_architecture.py   GET  /api/model_architecture (PNG)
api/execute.py              POST /api/execute            (main entry)
agent/llm.py                LLMod client via the OpenAI SDK (same as your RAG code)
agent/tools.py              tool implementations
agent/agent.py              orchestrator + step tracing (this is where the prompts live)
index.html                  GUI, served statically at /
architecture.png            served by /api/model_architecture
scripts/make_architecture.py  regenerates the PNG (needs matplotlib)
```
Each `api/*.py` uses Vercel's native `BaseHTTPRequestHandler` pattern (same as your RAG `prompt.py`).
No Flask, no rewrites: `/api/*` map to the functions automatically and `index.html` is served at `/`.

## Endpoints
- `GET /` — GUI
- `GET /api/team_info`
- `GET /api/agent_info`
- `GET /api/model_architecture` (PNG)
- `POST /api/execute` — body `{ "prompt": "..." }` → `{ status, error, response, steps }`

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env .env                                  # fill LLMOD_API_KEY, set base_url + model
vercel dev                                            # serves GUI + /api/* like production
```
(`agent/llm.py` calls `load_dotenv()`, so `.env` is picked up automatically.)

## Deploy
1. Push this folder to a GitHub repo.
2. vercel.com → New Project → import the repo (Python auto-detected via requirements.txt).
3. Settings → Environment Variables: add `LLMOD_API_KEY`, `LLMOD_BASE_URL`, `LLMOD_MODEL`.
4. Deploy, then fill your real details in `api/team_info.py`. Submit Vercel URL + GitHub URL.

## Where the "skills" (prompt engineering) live
There is no training/fine-tuning. Each module is the SAME model with a different **system prompt**.
Tune a skill by editing its `sys = "..."` string in `agent/agent.py`:
`_profile` (Preference Profiler), `_plan` (ReAct Planner), `_reflect` (Reflection Layer),
`_format` (Output Formatter). The `/api/agent_info` `prompt_template` is the user-facing template.

## Notes
- `LLMOD_BASE_URL` / `LLMOD_MODEL` — reuse the exact values from your RAG assignment's config.
- Vercel Hobby caps function duration at 60s (Pro: 300s). A run is ~15–40s; if you hit timeouts on
  Hobby, lower `MAX_PLANNER_STEPS` in `agent/agent.py`.
- Budget: ReAct loop capped at 6 tool calls + reflection at 2 cycles → ~6–9 LLM calls/request.
  Tools other than weather cost nothing.
- If `/api/model_architecture` 500s in production, confirm `includeFiles` in `vercel.json` ships
  `architecture.png`. If any function can't import `agent`, add `"includeFiles": "agent/**"` to it.
- To make `maps_tool`/`search_tool`/`reviews_tool` live, replace the mock bodies in `agent/tools.py`
  (marked `SWAP:`) with Google Places / SerpAPI / TripAdvisor calls — or back them with Pinecone
  the same way your RAG assignment does.
