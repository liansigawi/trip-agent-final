"""GET /api/agent_info — agent meta + how to use it."""
import json
from http.server import BaseHTTPRequestHandler

INFO = {
    "description":
        "An autonomous Trip Planning agent. From a single natural-language request it profiles "
        "the traveller, runs a ReAct loop over travel tools (live weather + maps/search/reviews, "
        "plus fictive flights/booking), self-critiques the draft, and returns a costed day-by-day "
        "itinerary.",
    "purpose":
        "Replace hours of fragmented trip research with one autonomous pass that produces a "
        "personalized, budget-aware, geographically sane itinerary.",
    "prompt_template": {
        "template":
            "Plan a {days}-day trip to {destination} for a {group} who likes {style}. "
            "Budget: {budget}. Must-see: {priorities}. Avoid: {avoid}. "
            "Accessibility needs: {accessibility}.",
    },
    "prompt_examples": [{
        "prompt": "7 days in Japan, a couple, mid-range budget, love food and culture, "
                  "must see Kyoto temples and Mt Fuji.",
        "full_response": "A 7-day Japan itinerary (Markdown), day-by-day with times, venue tips, "
                         "durations and per-day + grand-total costs.",
        "steps": [
            "Preference Profiler — extract typed profile.",
            "ReAct Planner — Thought/Action/Observation loop; weather_tool (live), maps_tool, reviews_tool.",
            "Reflection Layer — critic checks geography, time, budget, hours; PASS or re-plan.",
            "Output Formatter — render validated plan into a costed itinerary.",
        ],
    }],
}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._respond(200, INFO)

    def do_OPTIONS(self):
        self._respond(204, None)

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if data is not None:
            self.wfile.write(json.dumps(data).encode())
