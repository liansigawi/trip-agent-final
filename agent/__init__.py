"""Trip Agent core package (planner, tools, LLM client, schemas, observability).

Marks `agent` as a regular package so `from agent.agent import run_agent` resolves
reliably inside Vercel's bundled serverless functions (not just via implicit
namespace-package resolution)."""
