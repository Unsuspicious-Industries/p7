import aufbau

from proposition7.agentic import Tool, ToolRegistry, Param

_FOUR_TOOL_REGISTRY = ToolRegistry(
    [
        Tool.unary("search", "string", "docs", lambda arg: f"docs_about_{arg}"),
        Tool.unary("summarize", "docs", "string", lambda arg: f"summary_of_{arg}"),
        Tool.unary("count", "docs", "int", lambda arg: 3),
        Tool.unary("format", "int", "string", lambda arg: f"formatted_{arg}"),
    ]
)

_MULTI_ARG_REGISTRY = ToolRegistry(
    [
        Tool.nullary("clock", "int", lambda: 1000),
        Tool("add", (Param("a", "int"), Param("b", "int")), "int", lambda a, b: a + b),
        Tool.unary("to_string", "int", "string", lambda a: f"n{a}"),
    ]
)


def _status(spec: str, program: str) -> str:
    spg = aufbau.SPG(spec)
    return aufbau.Synthesizer.from_grammar(spg, program).status()


def test_generated_spec_compiles_for_every_root_and_syntax_combination():
    for root in ("program", "step"):
        for syntax in ("let_c", "sexpr"):
            aufbau.SPG(_FOUR_TOOL_REGISTRY.spec(root=root, syntax=syntax))


def test_let_c_program_accepts_well_typed_pipeline():
    spec = _FOUR_TOOL_REGISTRY.spec(root="program", syntax="let_c")
    program = 'let r = search("agents"); let s = summarize(r); return s;'
    assert _status(spec, program) == "typed"


def test_let_c_program_rejects_type_mismatched_pipeline():
    spec = _FOUR_TOOL_REGISTRY.spec(root="program", syntax="let_c")
    program = 'let r = search("agents"); let n = format(r); return n;'
    assert _status(spec, program) == "dead"


def test_sexpr_program_accepts_well_typed_pipeline():
    spec = _FOUR_TOOL_REGISTRY.spec(root="program", syntax="sexpr")
    program = '(let r (search "agents")) (let s (summarize r)) (return s)'
    assert _status(spec, program) == "typed"


def test_step_root_supports_incremental_context():
    spec = _FOUR_TOOL_REGISTRY.spec(root="step", syntax="let_c")
    spg = aufbau.SPG(spec)
    synth = aufbau.Synthesizer.from_grammar(spg, "")
    synth.add_to_ctx("r", "docs")
    synth.set_input("let s = summarize(r);")
    assert synth.status() == "typed"


def test_nullary_and_multi_arg_tools_accept_well_typed_pipeline():
    for syntax, program in [
        ("let_c", 'let t = clock(); let s = add(t, 5); let r = to_string(s); return r;'),
        ("sexpr", "(let t (clock)) (let s (add t 5)) (let r (to_string s)) (return r)"),
    ]:
        spec = _MULTI_ARG_REGISTRY.spec(root="program", syntax=syntax)
        assert _status(spec, program) == "typed"


def test_multi_arg_tool_rejects_wrong_argument_type():
    for syntax, program in [
        ("let_c", 'let t = clock(); let s = add(t, "x"); return s;'),
        ("sexpr", '(let t (clock)) (let s (add t "x")) (return s)'),
    ]:
        spec = _MULTI_ARG_REGISTRY.spec(root="program", syntax=syntax)
        assert _status(spec, program) == "dead"


def test_registry_execute_dispatches_positionally():
    assert _FOUR_TOOL_REGISTRY.execute("search", ["x"]) == "docs_about_x"
    assert _MULTI_ARG_REGISTRY.execute("add", [2, 3]) == 5
    assert _MULTI_ARG_REGISTRY.execute("clock", []) == 1000


def test_registry_rejects_mismatched_key_and_tool_name():
    import pytest

    with pytest.raises(ValueError):
        ToolRegistry({"wrong_key": Tool.unary("search", "string", "docs", lambda a: a)})
