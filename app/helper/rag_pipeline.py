import re
from app.RagPipeline.state import AgentState
from typing import List
class RagPipeLineHelper:

    @staticmethod
    def _extract_first_word(text: str) -> str:
        match = re.search(r'\b([a-z_]+)\b', text.lower().strip())
        return match.group(1) if match else ""

    @staticmethod
    def _safe_steps(state: AgentState, step: str) -> List[str]:
        steps = list(state.get("steps") or [])
        steps.append(step)
        return steps

    @staticmethod
    def _is_bad_rewrite(original: str, rewritten: str) -> bool:
        refusal_signals = [
            "i don't know", "i cannot", "i'm unable", "as an ai",
            "i don't have", "cannot rewrite", "unclear question",
        ]
        if not rewritten or len(rewritten.strip()) < 5:
            return True
        if len(rewritten) > len(original) * 3:
            return True
        if any(sig in rewritten.lower() for sig in refusal_signals):
            return True
        return False

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for item in items:
            key = item.strip()[:200]
            if key and key not in seen:
                seen.add(key)
                out.append(item)
        return out