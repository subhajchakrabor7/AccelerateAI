"""Agent observability logger -- safe for Streamlit reruns.

Uses core.logger.log() instead of print() to avoid ValueError on Streamlit's
closed stdout handle.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from core.logger import log

TRACES_DIR = Path(__file__).resolve().parent.parent / "data" / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    """Collects observability data for a single agent invocation."""

    def __init__(self, agent_name: str, run_id: str):
        self.agent_name = agent_name
        self.run_id = run_id
        self.start_time = time.time()
        self.trace: dict = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": "",
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": 0.0,
            "status": "started",
        }

    def set_input(self, **kwargs) -> "AgentTrace":
        self.trace["input"] = dict(kwargs)
        return self

    def set_plan(self, plan: str) -> "AgentTrace":
        self.trace["plan"] = plan
        return self

    def set_output(self, **kwargs) -> "AgentTrace":
        self.trace["output"] = dict(kwargs)
        return self

    def extract_from_messages(self, messages: list) -> "AgentTrace":
        tool_calls: list = []
        reasoning_steps: list = []
        plan_text: str = ""

        for msg in messages:
            msg_type = type(msg).__name__
            content = getattr(msg, "content", "")

            if msg_type == "HumanMessage":
                if isinstance(content, str) and content.strip():
                    reasoning_steps.append({
                        "role": "task_input",
                        "content": content.strip()[:400],
                    })
            elif msg_type == "AIMessage":
                raw_tool_calls = getattr(msg, "tool_calls", []) or []
                for tc in raw_tool_calls:
                    tool_calls.append({
                        "tool": (
                            tc.get("name", "unknown") if isinstance(tc, dict)
                            else getattr(tc, "name", "unknown")
                        ),
                        "args": (
                            tc.get("args", {}) if isinstance(tc, dict)
                            else getattr(tc, "args", {})
                        ),
                    })
                if isinstance(content, str) and content.strip():
                    reasoning_steps.append({
                        "role": "ai_reasoning",
                        "content": content.strip()[:600],
                    })
                    if not plan_text and len(content.strip()) > 20:
                        plan_text = content.strip()[:600]
            elif msg_type == "ToolMessage":
                tool_name = getattr(msg, "name", "unknown_tool")
                raw = content if isinstance(content, str) else str(content)
                preview = raw[:400] + "..." if len(raw) > 400 else raw
                reasoning_steps.append({
                    "role": "tool_result",
                    "tool": tool_name,
                    "content": preview,
                })

        self.trace["tool_calls"] = tool_calls
        self.trace["reasoning_steps"] = reasoning_steps
        if plan_text and not self.trace["plan"]:
            self.trace["plan"] = plan_text
        return self

    def complete(self, status: str = "success") -> dict:
        self.trace["duration_seconds"] = round(time.time() - self.start_time, 3)
        self.trace["status"] = status
        self._save()
        log(
            f"[OBSERVE][{self.agent_name}] status={self.trace['status']} "
            f"duration={self.trace['duration_seconds']}s "
            f"tools_called={[t['tool'] for t in self.trace['tool_calls']]}"
        )
        return self.trace

    def fail(self, error: str) -> dict:
        self.trace["error"] = error
        return self.complete(status="failed")

    def _save(self):
        path = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
        existing: list = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                existing = data if isinstance(data, list) else [data]
            except Exception:
                existing = []
        existing.append(self.trace)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, default=str)
