"""tools/llamaguard_sensitivity.py: Llama Guard output parsing never coerces, and kappa is the textbook formula."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import llamaguard_sensitivity as lgs                            # noqa: E402


def test_parse_reads_llama_guard_3_output():
    assert lgs.parse("safe") == (0, [])
    assert lgs.parse("\n\nunsafe\nS1") == (1, ["S1"])
    assert lgs.parse("unsafe\nS2,S10") == (1, ["S2", "S10"])
    assert lgs.parse("Unsafe S9") == (1, ["S9"])


def test_parse_never_coerces_an_unexpected_answer():
    assert lgs.parse("") == (None, [])
    assert lgs.parse("I cannot classify this.") == (None, [])
    assert lgs.parse("The response is safe") == (None, [])      # not the model's format: unparsed, not 0


def test_amendment_a1_prompt_assesses_the_agent_turn_and_keeps_its_own_store():
    p = lgs.AGENT_PROMPT.format(prompt="Q", reply="A")
    assert "ONLY THE LAST Agent message" in p and "User message" not in p
    assert "User: Q\n\nAgent: A" in p and p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert lgs.CALLS["agent"]["tag"] != lgs.CALLS["chat"]["tag"]
    assert lgs.store_path("sciencesafety", "agent") != lgs.store_path("sciencesafety", "chat")
    assert lgs.INSTRUMENT["chat"]["status"] == "FAILED"


def test_kappa():
    assert lgs.kappa(20, 0, 0, 20) == 1.0
    assert lgs.kappa(10, 10, 10, 10) == 0.0
    assert lgs.kappa(0, 0, 0, 0) is None
    assert abs(lgs.kappa(45, 15, 25, 15) - 0.1304) < 1e-3       # po 0.60, pe (60*70 + 40*30) / 100^2 = 0.54
