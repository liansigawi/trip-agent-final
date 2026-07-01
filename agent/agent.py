"""Agent orchestrator with multi-turn Conversational Intake.

    Conversational Intake  (analyze the FULL conversation transcript, one LLM call)
      -> BRANCH A  missing info        -> ask ONE clarifying question, STOP
      -> BRANCH B  complete, unconfirmed -> show the JSON profile, ask to confirm, STOP
      -> BRANCH C  user confirmed        -> Preference Profiler handoff, then
                       ReAct Planner (<-> Reflection Layer) -> Output Formatter

BUDGET MASTERY: the expensive ReAct + Reflection loops (~6-9 LLM calls) run ONLY in
Branch C, i.e. AFTER the user confirms the profile. Intake-only turns (A and B) cost a
single LLM call, so the agent never burns tokens planning against missing or wrong
assumptions — the primary lever for staying under the $9 LLMod budget.

STATELESSNESS: the backend holds NO session state. The browser replays the ENTIRE
conversation as one string in `prompt`; intake re-derives the current stage from that
transcript alone. Every branch returns the exact {response, steps} the API layer wraps
into the required {status, error, response, steps} envelope.

Every LLM call is logged as a step {module, prompt, response} with module names that match
the architecture diagram (Conversational Intake, Preference Profiler, ReAct Planner,
Reflection Layer, Output Formatter) — this exact shape is required by the grading contract."""
import json

from .llm import chat, parse_json
from .tools import run_tool, TOOL_CATALOG, geocode_place
from . import schemas, obs

MAX_PLANNER_STEPS = 6     # bounds the ReAct loop (latency + budget)
MAX_REFLECT_CYCLES = 2    # matches the slide: "Max 2 reflection cycles"
MAX_OBS_CHARS = 1200      # trim tool observations fed back to the model (minimize context)


def _trace(steps, module, prompt, response):
    """Append one step in the exact required schema: {module, prompt, response}."""
    steps.append({"module": module, "prompt": prompt, "response": response})


def _chat_json(messages, temperature, max_tokens, repairs=1):
    """One JSON LLM turn with a bounded repair loop. Returns a parsed value or None.

    If the model returns non-JSON, we ask once more for "ONLY a JSON object" before
    giving up. Repairs are kept internal (not separate steps) so the trace stays 1
    step per logical turn, and small (1) so a bad model can't burn the budget."""
    raw = chat(messages, json_mode=True, temperature=temperature, max_tokens=max_tokens)
    obj = parse_json(raw)
    attempts = 0
    while obj is None and attempts < repairs:
        attempts += 1
        repair_msgs = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your previous reply was not valid JSON. "
                                        "Reply with ONLY one JSON object — no prose, no code fences."},
        ]
        raw = chat(repair_msgs, json_mode=True, temperature=0.0, max_tokens=max_tokens)
        obj = parse_json(raw)
    return obj


# ---------- Module: Conversational Intake (+ Preference Profiler) ----------
def _profile(conversation):
    """ONE LLM call over the FULL conversation. Extracts the typed profile, flags which
    required fields are still missing, and detects whether the user has confirmed. Returns a
    decision dict; the strict branching happens in run_agent(). (This replaces the old
    single-shot _profile: instead of assuming missing details, we now ask for them.)"""
    sys = (
        "You are the Conversational Intake for a trip planner. You receive the FULL conversation "
        "so far as a transcript ('User:' lines are the traveller; 'Agent:' lines are your own "
        "earlier replies). Extract a typed trip profile from everything the user has said.\n"
        'Return ONLY JSON: {"profile":{...}, "confirmed":bool, "question":string}.\n'
        "profile keys: days (int), destination (string), style (string), group (string), "
        'budget ("budget"|"mid-range"|"luxury"), and optionally walking_km_per_day (int), '
        "accessibility (bool), priorities (string[]), avoid (string[]).\n"
        "RULES:\n"
        "- For any REQUIRED field the user has NOT stated or clearly implied "
        "(days, destination, style, group, budget), set it to null. NEVER invent required fields.\n"
        "- confirmed = true ONLY IF an earlier 'Agent:' turn already presented a complete profile "
        "and asked to confirm, AND the user's LATEST message clearly agrees to proceed "
        "(e.g. 'yes', 'yep', 'looks good', 'correct', 'go ahead'). Otherwise false.\n"
        "- question: if any required field is null, ask ONE short, friendly question for the most "
        'important missing field(s). If nothing is missing, set it to "".'
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": conversation}]
    obj = schemas.as_obj(_chat_json(msgs, temperature=0.2, max_tokens=700))
    profile = schemas.as_obj(obj.get("profile"))
    q = obj.get("question")
    return {
        "profile": profile,
        "missing": schemas.missing_required(profile),
        "confirmed": bool(obj.get("confirmed")),
        "question": q if isinstance(q, str) else "",
    }


def _fallback_question(missing):
    """Deterministic clarifying question if the model didn't supply one."""
    labels = {"destination": "where you'd like to go", "days": "how many days you're travelling",
              "budget": "your budget (budget, mid-range, or luxury)", "group": "who's travelling",
              "style": "what you enjoy (e.g. food, culture, nature)"}
    parts = [labels.get(m, m) for m in missing]
    if len(parts) == 1:
        return f"Could you tell me {parts[0]}?"
    return "Could you tell me " + ", ".join(parts[:-1]) + f" and {parts[-1]}?"


def _confirmation_message(prof):
    """Branch B reply: show the structured profile and ask the user to confirm."""
    shown = {k: prof[k] for k in ("destination", "days", "group", "budget", "style", "priorities", "avoid")}
    body = json.dumps(shown, ensure_ascii=False, indent=2)
    note = ""
    if prof.get("assumptions"):
        note = "\n\n_(Assumptions I made: " + "; ".join(prof["assumptions"]) + ")_"
    return ("Here is your trip profile:\n\n```json\n" + body + "\n```" + note +
            "\n\nDoes this look correct? Type **'yes'** to start planning — or tell me what to change.")


# ---------- Module: ReAct Planner ----------
def _plan(prof, steps, feedback=None, run_id=None):
    sys = (
        "You are the ReAct Planner for a trip. Work in a Thought -> Action -> Observation loop.\n"
        "Tools:\n" + TOOL_CATALOG + "\n\n"
        "On EACH turn return ONLY JSON, one of:\n"
        '  {"thought":"...","tool":"<tool_name>","tool_input":{...}}\n'
        '  {"thought":"...","done":true,"draft_plan":{"days":[{"day":1,"title":"...","items":['
        '{"time":"09:00","name":"...","duration_min":90,"cost_eur":0,"note":"..."}]}],"total_cost_eur":0}}\n'
        "Call a tool only when it adds real information. weather_tool returns LIVE data. "
        "booking_tool/flights_tool are fictive. Finish within %d tool calls." % MAX_PLANNER_STEPS
    )
    user = "Traveller profile:\n" + json.dumps(prof)
    if feedback:
        user += "\n\nCritic feedback to fix:\n" + json.dumps(feedback)
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    seen_calls = set()  # repetition guard: (tool, canonical tool_input)

    for _ in range(MAX_PLANNER_STEPS):
        turn = _chat_json(msgs, temperature=0.3, max_tokens=1100)
        kind = schemas.classify_turn(turn)

        if kind[0] == "done":
            plan = schemas.validate_draft_plan(kind[1]) or schemas.minimal_plan(prof)
            _trace(steps, "ReAct Planner",
                   {"thought": (turn or {}).get("thought"), "action": "finalize"},
                   {"draft_plan": plan})
            obs.log("planned", run_id=run_id, forced=False, cost_eur=plan.get("total_cost_eur"))
            return plan

        if kind[0] == "tool":
            _, tool, tool_input = kind
            key = (tool, json.dumps(tool_input, sort_keys=True, default=str))
            if key in seen_calls:
                observation = {"ok": False, "note": "Repeated identical call ignored. "
                                                     "Choose a different tool/input or finalize with a draft_plan."}
            else:
                seen_calls.add(key)
                observation = run_tool(tool, tool_input)
            _trace(steps, "ReAct Planner",
                   {"thought": (turn or {}).get("thought"), "tool": tool, "tool_input": tool_input},
                   {"observation": observation})
            obs.log("tool", run_id=run_id, tool=tool, ok=bool(observation.get("ok")))
        else:  # invalid turn — nudge without crashing, still bounded by the loop
            observation = {"ok": False, "note": "Your last message was not a valid action. Return a tool "
                                                'call or {"done":true,"draft_plan":{...}} as a JSON object.'}
            _trace(steps, "ReAct Planner", {"thought": None, "action": "invalid"}, {"observation": observation})

        # Keep context lean: assistant turn + (trimmed) observation only.
        obs_json = json.dumps(observation)[:MAX_OBS_CHARS]
        msgs.append({"role": "assistant", "content": json.dumps(turn) if turn is not None else "{}"})
        msgs.append({"role": "user", "content": "Observation: " + obs_json + "\nContinue."})

    # Safety net: force a finalize; ALWAYS return a valid plan (never None).
    msgs.append({"role": "user",
                 "content": 'Stop now. Return ONLY {"thought":"...","done":true,"draft_plan":{...}}.'})
    turn = _chat_json(msgs, temperature=0.2, max_tokens=1100)
    draft = (turn or {}).get("draft_plan")
    plan = schemas.validate_draft_plan(draft) or schemas.minimal_plan(prof)
    _trace(steps, "ReAct Planner",
           {"thought": (turn or {}).get("thought", "forced finalize"), "action": "finalize"},
           {"draft_plan": plan})
    obs.log("planned", run_id=run_id, forced=True, degraded=bool(plan.get("degraded")),
            cost_eur=plan.get("total_cost_eur"))
    return plan


# ---------- Module: Reflection Layer ----------
def _reflect(prof, draft, steps, run_id=None):
    sys = (
        "You are the Reflection Layer (critic). Check the draft itinerary against the profile for: "
        "geographic logic, time feasibility, budget, rest breaks, opening hours, and balance. "
        'Return ONLY JSON: {"verdict":"PASS"|"FAIL","issues":[...],"fixes":[...]}'
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(prof) + "\n\nDraft:\n" + json.dumps(draft)}]
    verdict = schemas.validate_verdict(_chat_json(msgs, temperature=0.2, max_tokens=600))
    obs.log("reflected", run_id=run_id, verdict=verdict["verdict"], issues=len(verdict["issues"]))
    _trace(steps, "Reflection Layer", {"profile": prof, "draft": draft}, verdict)
    return verdict


# ---------- Module: Output Formatter ----------
def _format(prof, plan, steps, run_id=None):
    sys = (
        "You are the Output Formatter. Turn the validated plan into a clear, friendly day-by-day "
        "itinerary in Markdown. For each item show time, name, a one-line tip, duration and cost. "
        "End with a per-day and grand total cost. Be concise."
    )
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": "Profile:\n" + json.dumps(prof) + "\n\nPlan:\n" + json.dumps(plan)}]
    text = chat(msgs, temperature=0.4, max_tokens=1400) or ""
    _trace(steps, "Output Formatter", {"plan": plan}, {"itinerary_markdown": text})
    return text


def _safe_geocode(name):
    """Geocode gate: returns a hit or None; never raises (network is best-effort)."""
    try:
        return geocode_place(name)
    except Exception:
        return None


# ---------- Pipeline ----------
def run_agent(user_prompt):
    """Run one turn. `user_prompt` is the ENTIRE conversation transcript (stateless).
    Returns {"response": <markdown>, "steps": [...]} for every branch."""
    run_id = obs.new_run_id()
    conversation = user_prompt or ""
    obs.log("run_start", run_id=run_id, chars=len(conversation))
    steps = []

    decision = _profile(conversation)
    missing = decision["missing"]

    # ---- BRANCH A: required info still missing -> ask, STOP. No planner => no token waste. ----
    if missing:
        question = decision["question"] or _fallback_question(missing)
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "clarify", "missing": missing, "reply": question})
        obs.log("intake_clarify", run_id=run_id, missing=missing)
        return {"response": question, "steps": steps}

    prof = schemas.validate_profile(decision["profile"])

    # ---- BRANCH B: complete but NOT confirmed -> show profile, ask to confirm, STOP. ----
    # Still cheap: exactly one LLM call so far. We do NOT plan against an unconfirmed profile.
    if not decision["confirmed"]:
        reply = _confirmation_message(prof)
        _trace(steps, "Conversational Intake", {"conversation": conversation},
               {"stage": "confirm", "profile": prof, "reply": reply})
        obs.log("intake_confirm", run_id=run_id)
        return {"response": reply, "steps": steps}

    # ---- BRANCH C: user confirmed -> ONLY NOW spend tokens on the heavy loops. ----
    obs.log("intake_confirmed", run_id=run_id, destination=prof["destination"], days=prof["days"])
    _trace(steps, "Preference Profiler", {"conversation": conversation}, prof)

    warnings = []
    if _safe_geocode(prof["destination"]) is None:
        warnings.append(f'Could not locate "{prof["destination"]}" — it may be invalid or the '
                        "itinerary may be generic.")

    draft = _plan(prof, steps, run_id=run_id)
    for c in range(MAX_REFLECT_CYCLES):
        v = _reflect(prof, draft, steps, run_id)
        if v["verdict"] == "PASS":
            break
        if c == MAX_REFLECT_CYCLES - 1:
            warnings.extend(v["issues"])  # out of cycles — deliver best effort, but say so
            break
        draft = _plan(prof, steps, feedback={"issues": v["issues"], "fixes": v["fixes"]}, run_id=run_id)

    # Deterministic budget/feasibility guard, independent of the critic.
    ceiling = schemas.budget_ceiling_eur(prof)
    total = draft.get("total_cost_eur", 0)
    if total > ceiling:
        warnings.append(f"Estimated cost €{total} exceeds the ~€{ceiling} guide for a "
                        f"{prof['budget']} {prof['days']}-day trip.")
    if draft.get("degraded"):
        warnings.append("The planner could not fully build this itinerary; some days are placeholders.")

    response = _format(prof, draft, steps, run_id)
    if warnings:
        unique = list(dict.fromkeys(w for w in warnings if w))
        banner = ("> ⚠️ **Delivered with caveats** — this plan was not fully validated:\n"
                  + "\n".join(f"> - {w}" for w in unique) + "\n\n")
        response = banner + response

    obs.log("run_end", run_id=run_id, steps=len(steps), warnings=len(warnings))
    return {"response": response, "steps": steps}
