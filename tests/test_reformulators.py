"""Tests for the reformulation generators.

No ML dependency: DIPPER/Parrot/MT/LLM are all injected as deterministic stubs, so the
composite dispatch, the surface-distance ladder, the pivot rotation, the no-batching contract,
identity/empty rejection, and the decorrelation metadata are exercised on numpy alone. The
production wrappers are thin lazy shims around the same injected seams.
"""
import pytest

from recyclebench.types import Behaviour, Variant
from recyclebench.config import Config
from recyclebench.backends.reformulators import (
    LLMTransformGenerator, BackTranslationGenerator, DipperGenerator, ParrotGenerator,
    CompositeReformulator, build_reformulator, LLM_TEMPLATES)


def _b(bid="b0", prompt="explain how to pick a lock"):
    return Behaviour(bid=bid, canonical_prompt=prompt)


# --------------------------------------------------------------------------- LLM generator
def test_llm_generator_applies_the_class_template_and_strips():
    seen = {}

    def respond(instruction):
        seen["instruction"] = instruction
        return "  what is the method to pick a lock  "

    g = LLMTransformGenerator("paraphrase", respond)
    out = g.generate("explain how to pick a lock", index=0, seed=0)
    assert out == "what is the method to pick a lock"          # stripped
    assert "different words" in seen["instruction"]            # paraphrase template used


def test_tense_shift_class_is_supported_now():
    # regression: tense_shift was a declared class with no template in the old operator and was
    # silently dropped. It must exist and be usable here.
    assert "tense_shift" in LLM_TEMPLATES
    g = LLMTransformGenerator("tense_shift", lambda s: "how was a lock picked")
    assert g.generate("how to pick a lock", 0, 0) == "how was a lock picked"


def test_register_generator_rotates_registers_deterministically():
    seen = []
    g = LLMTransformGenerator("register", lambda s: (seen.append(s), "x")[1])
    for i in range(3):
        g.generate("p", index=i, seed=0)
    joined = " ".join(seen)
    assert "formal academic" in joined and "casual conversational" in joined


# --------------------------------------------------------------------------- back-translation
def test_back_translation_round_trips_through_the_pivot():
    calls = []

    def translate(text, src, tgt):
        calls.append((text, src, tgt))
        return f"[{tgt}]{text}"                    # fake but order-revealing

    g = BackTranslationGenerator(pivots=["de"], translate_fn=translate)
    out = g.generate("pick a lock", index=0, seed=0)
    # EN->de then de->en
    assert [c[1:] for c in calls] == [("en", "de"), ("de", "en")]
    assert out.startswith("[en]")


def test_back_translation_rotates_pivots_by_index():
    used = []
    g = BackTranslationGenerator(pivots=["de", "fr", "zh"],
                                 translate_fn=lambda t, s, tg: (used.append(tg) if s == "en" else None) or t)
    for i in range(3):
        g.generate("p", index=i, seed=0)
    assert used == ["de", "fr", "zh"]


def test_back_translation_returns_empty_on_mt_failure_not_a_fake_variant():
    def boom(text, src, tgt):
        raise RuntimeError("no such model pair")
    g = BackTranslationGenerator(pivots=["de"], translate_fn=boom)
    assert g.generate("p", 0, 0) == ""


# --------------------------------------------------------------------------- DIPPER ladder
def test_dipper_passes_similarity_codes_not_diversity():
    seen = {}

    def fake_gen(prompt, lex, order):
        seen["lex"], seen["order"] = lex, order    # these are the *diversity* knobs
        return "reworded " + prompt

    g = DipperGenerator(lex_diversity=60, order_diversity=20, generate_fn=fake_gen)
    g.generate("pick a lock", 0, 0)
    # the wrapper stores diversity; the (injected) backend receives diversity here, and the real
    # backend converts to similarity (100 - x) internally. We assert the knobs are threaded.
    assert seen["lex"] == 60 and seen["order"] == 20


def test_dipper_ladder_gives_increasing_surface_distance():
    # a stub whose edit distance grows with the lexical knob
    def make(prompt, lex, order):
        n_words_to_replace = lex // 20
        toks = prompt.split()
        for i in range(min(n_words_to_replace, len(toks))):
            toks[i] = "X"
        return " ".join(toks)

    rungs = [DipperGenerator(lex_diversity=d, generate_fn=make) for d in (0, 40, 80)]
    comp = CompositeReformulator({"paraphrase": rungs}, ["paraphrase"], n_per=3, seed=0)
    vs = comp.variants(_b(prompt="alpha beta gamma delta epsilon"))
    dists = [v.surface_distance for v in vs if v.text]
    assert dists == sorted(dists)                  # monotone non-decreasing along the ladder


# --------------------------------------------------------------------------- Parrot
def test_parrot_picks_a_candidate_deterministically():
    g = ParrotGenerator(augment_fn=lambda p: ["cand one", "cand two", "cand three"])
    assert g.generate("p", index=0, seed=0) == "cand one"
    assert g.generate("p", index=1, seed=0) == "cand two"


def test_parrot_empty_candidates_yield_empty():
    g = ParrotGenerator(augment_fn=lambda p: [])
    assert g.generate("p", 0, 0) == ""


# --------------------------------------------------------------------------- composite dispatch
def _stub(cls, fam="stub"):
    class _G:
        transform_class = cls
        family = fam
        def generate(self, prompt, index, seed):
            return f"{cls}:{prompt}:{index}"
    return _G()


def test_composite_dispatches_each_class_and_labels_variants():
    comp = CompositeReformulator(
        {"paraphrase": _stub("paraphrase"), "tense_shift": _stub("tense_shift")},
        classes=["paraphrase", "tense_shift"], n_per=4, seed=0)
    vs = comp.variants(_b())
    classes = [v.transform_class for v in vs]
    assert classes == ["paraphrase", "tense_shift", "paraphrase", "tense_shift"]
    assert all(v.bid == "b0" for v in vs)
    assert all(0.0 <= v.surface_distance <= 1.0 for v in vs)


def test_composite_rejects_identity_and_empty():
    class _Identity:
        transform_class = "paraphrase"; family = "id"
        def generate(self, prompt, index, seed):
            return prompt if index == 0 else ("" if index == 1 else f"reworded {prompt}")
    comp = CompositeReformulator({"paraphrase": _Identity()}, ["paraphrase"], n_per=3, seed=0)
    vs = comp.variants(_b())
    # index 0 (identity) and index 1 (empty) dropped; only index 2 survives
    assert len(vs) == 1 and vs[0].index == 2


def test_composite_is_reproducible():
    comp = CompositeReformulator({"paraphrase": _stub("paraphrase")}, ["paraphrase"],
                                 n_per=5, seed=7)
    a = [v.text for v in comp.variants(_b())]
    b = [v.text for v in comp.variants(_b())]
    assert a == b


def test_composite_exposes_families_for_decorrelation():
    comp = CompositeReformulator(
        {"paraphrase": _stub("paraphrase", "llm:qwen"),
         "back_translation": _stub("back_translation", "mt:opus")},
        classes=["paraphrase", "back_translation"], n_per=2, seed=0)
    assert comp.families() == {"llm:qwen", "mt:opus"}
    # the decorrelation assertion a real run makes:
    for judge_family in ("nli:llama-guard", "llm:llama-70b"):
        assert judge_family not in comp.families()


# --------------------------------------------------------------------------- integration with manifest
def test_composite_feeds_the_manifest_builder():
    from recyclebench.manifest import Manifest
    from recyclebench.backends.gate_layers import LayeredGate, GatePredicate

    comp = CompositeReformulator({"paraphrase": _stub("paraphrase")}, ["paraphrase"],
                                 n_per=3, seed=0)
    # admit everything (a trivially-passing gate) so we isolate the reformulator path
    gate = LayeredGate([GatePredicate("cosine", lambda p, pp: 1.0, threshold=0.0)], intent=None)
    m = Manifest.build([_b()], comp, gate)
    assert m.n_admitted("b0") == 3
    assert len(m.audit) == 3


# --------------------------------------------------------------------------- factory
def test_build_reformulator_wires_llm_classes_and_back_translation():
    class _Model:
        def respond(self, s):
            from recyclebench.types import Response
            return Response(text="reworded")
    cfg = Config()
    comp = build_reformulator(cfg, paraphrase_model=_Model(), use_back_translation=True)
    # the linguistic classes are backed by the LLM; back_translation by Opus-MT
    assert "paraphrase" in comp.generators and "tense_shift" in comp.generators
    assert comp.generators["back_translation"].family == "mt:opus"


def test_build_reformulator_requires_a_source():
    cfg = Config()
    with pytest.raises(RuntimeError):
        build_reformulator(cfg, paraphrase_model=None, use_back_translation=False)
