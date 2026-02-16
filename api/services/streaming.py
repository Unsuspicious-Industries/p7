from __future__ import annotations

import json
from typing import Any, Dict


def sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"
