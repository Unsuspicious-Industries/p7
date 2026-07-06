"""Real-compiler oracle for the `c` grammar.

The `c` grammar's concrete syntax is real, compilable C, so a genuine C
compiler is a trusted external oracle: anything aufbau accepts must also be
something `cc` accepts. This is the C analogue of aufbau's own OCaml
differential certification harness (`ocaml/cert.ml` + `ocaml/oracle.ml`).
"""

from __future__ import annotations

import shutil
import subprocess

_COMPILER = shutil.which("cc") or shutil.which("gcc")


def compiles(source: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Check whether `source` is valid, compilable C via `cc -fsyntax-only`.

    Returns `(True, "")` when the compiler accepts the program, or
    `(False, reason)` with a short failure reason otherwise. Never raises:
    a missing compiler or a hung/crashed subprocess is reported as a failure,
    not an exception, so the oracle can never crash a benchmark run.
    """
    if _COMPILER is None:
        return False, "no_c_compiler_found"
    try:
        result = subprocess.run(
            [_COMPILER, "-fsyntax-only", "-xc", "-"],
            input=source,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "compiler_timeout"
    except OSError as error:
        return False, f"compiler_error:{error}"
    if result.returncode == 0:
        return True, ""
    return False, "compiler_rejected:" + " ".join(result.stderr.split())[:300]
