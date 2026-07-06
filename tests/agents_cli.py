from types import SimpleNamespace

import aufbau

from proposition7.agents import AgentSession
from proposition7.agents.cli import default_tools


def _fake_think(**kwargs):
    return SimpleNamespace(text="", tokens_generated=0)


def _scripted(spg, steps):
    def generate(prompt, initial, max_tokens, seed, context):
        synth = aufbau.Synthesizer.from_grammar(spg, "")
        for name, ty in context.items():
            synth.add_to_ctx(name, ty)
        text = steps[generate.i]
        generate.i += 1
        try:
            synth.set_input(text)
            status = synth.status()
        except Exception:
            status = "dead"
        return SimpleNamespace(text=text, is_complete=(status == "typed"), stopped_reason=status)

    generate.i = 0
    return generate


def test_default_tools_compile_and_execute_for_real():
    registry = default_tools()
    assert registry.execute("add", [2, 3]) == 5
    assert registry.execute("upper", ["hi"]) == "HI"
    assert registry.execute("length", ["hello"]) == 5
    assert isinstance(registry.execute("now", []), int)


def test_cli_registry_drives_a_real_episode_end_to_end():
    registry = default_tools()
    spg = aufbau.SPG(registry.spec(root="step", syntax="let_c"))
    steps = ['let s = upper("hi");', "return s;"]
    session = AgentSession(model=None, registry=registry, task="uppercase hi", max_turns=6)
    result = session.run_to_completion(think=_fake_think, generate=_scripted(spg, steps))

    assert result.success is True
    assert result.return_value == "HI"
