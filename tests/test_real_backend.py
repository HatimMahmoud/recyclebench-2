"""Conformance tests for the real backends.

``providers.py`` is written against the OpenAI chat-completions contract that vLLM also
serves. Rather than assume it works, these tests stand up a real HTTP server implementing
that contract and drive the adapter through it -- so the request shape, seed plumbing,
retry path, and cache all get exercised without a GPU or any external network.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from recyclebench.config import Config, make_rng
from recyclebench.types import Behaviour
from recyclebench.backends.providers import ProviderConfig, ProviderModel
from recyclebench.realrun import ResponseCache, CachedModel, pilot_p_minus
from recyclebench.judge import RuleBasedJudge


# ---------------------------------------------------------------------------
# A minimal OpenAI-compatible server (what `vllm serve` exposes)
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    received = []
    fail_times = 0          # transient failures to inject before succeeding
    refuse = False

    def log_message(self, *a):      # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        type(self).received.append(body)

        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"temporarily unavailable"}')
            return

        prompt = body["messages"][-1]["content"]
        seed = body.get("seed")
        text = ("I'm sorry, I can't help with that." if type(self).refuse
                else f"OK::{prompt[:24]}::seed={seed}")
        payload = {"id": "cmpl-1", "object": "chat.completion", "model": body.get("model"),
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text}}]}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    _Handler.received = []
    _Handler.fail_times = 0
    _Handler.refuse = False
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", _Handler
    httpd.shutdown()


def _model(url, name="llama-test"):
    return ProviderModel(name, ProviderConfig(base_url=url, model_id="meta-llama/Test",
                                              temperature=0.0, max_tokens=64,
                                              api_key_env="RECYCLEBENCH_TEST_KEY"))


openai = pytest.importorskip("openai", reason="openai SDK not installed")


# ---------------------------------------------------------------------------
def test_adapter_speaks_the_chat_completions_contract(server):
    url, H = server
    r = _model(url).respond("hello there")
    assert r.text.startswith("OK::hello there")
    assert r.latent_harmful is None          # real backends carry no latent truth
    sent = H.received[-1]
    assert sent["model"] == "meta-llama/Test"
    assert sent["messages"][-1]["role"] == "user"
    assert sent["temperature"] == 0.0


def test_seed_is_forwarded_so_k_sampling_is_k_draws(server):
    """At temperature 0 a real endpoint returns identical text unless the seed varies."""
    url, H = server
    m = _model(url)
    m.respond("p", seed=11)
    m.respond("p", seed=12)
    seeds = [b.get("seed") for b in H.received]
    assert seeds == [11, 12]


def test_transient_failures_are_retried(server):
    url, H = server
    H.fail_times = 2
    r = _model(url).respond("hello")          # should survive two 503s
    assert r.text.startswith("OK::")
    assert len(H.received) == 3


def test_permanent_failure_raises_rather_than_returning_empty(server):
    """A silent empty response would be judged as a refusal and quietly bias the estimate."""
    url, H = server
    H.fail_times = 99
    m = ProviderModel("x", ProviderConfig(base_url=url, model_id="m", max_retries=2,
                                          api_key_env="RECYCLEBENCH_TEST_KEY"))
    with pytest.raises(RuntimeError):
        m.respond("hello")


# ---------------------------------------------------------------------------
# cache / resumability
# ---------------------------------------------------------------------------
def test_cache_prevents_repaying_for_the_same_call(server, tmp_path):
    url, H = server
    cache = ResponseCache(tmp_path / "cache.jsonl")
    m = CachedModel(_model(url), cache)
    a = m.respond("same prompt", seed=1)
    b = m.respond("same prompt", seed=1)
    assert a.text == b.text
    assert m.calls == 1 and m.hits == 1
    assert len(H.received) == 1               # the endpoint was hit once


def test_cache_resumes_across_processes(server, tmp_path):
    url, H = server
    p = tmp_path / "cache.jsonl"
    m1 = CachedModel(_model(url), ResponseCache(p))
    m1.respond("resume me", seed=7)
    m1.cache.close()
    before = len(H.received)

    m2 = CachedModel(_model(url), ResponseCache(p))   # fresh cache object, same file
    r = m2.respond("resume me", seed=7)
    assert r.text.startswith("OK::resume me")
    assert m2.calls == 0 and m2.hits == 1
    assert len(H.received) == before                  # no new API call


def test_cache_distinguishes_seeds_and_models(server, tmp_path):
    url, _ = server
    cache = ResponseCache(tmp_path / "c.jsonl")
    m = CachedModel(_model(url, "model-a"), cache)
    n = CachedModel(_model(url, "model-b"), cache)
    m.respond("p", seed=1); m.respond("p", seed=2); n.respond("p", seed=1)
    assert len(cache) == 3          # same prompt, different seed/model -> distinct entries


def test_cache_can_be_disabled(tmp_path):
    c = ResponseCache(tmp_path / "off.jsonl", enabled=False)
    c.put("m", "p", 1, "text")
    assert c.get("m", "p", 1) is None
    assert not (tmp_path / "off.jsonl").exists()


# ---------------------------------------------------------------------------
# the real P^- pilot
# ---------------------------------------------------------------------------
def test_real_pilot_sizes_p_minus_without_touching_paraphrases(server):
    """The pilot must decide feasibility using canonical screening ONLY -- that is what makes
    it cheap enough to run before committing the paraphrase budget."""
    url, H = server
    cfg = Config(n_behaviours=0)
    cfg.budget.k_canonical_samples = 1
    behaviours = [Behaviour(bid=f"b{i}", canonical_prompt=f"request {i}") for i in range(6)]

    class Old:      # complies with everything -> breaks on all
        name = "old"
        def respond(self, prompt, *, behaviour=None, variant=None, canonical=False, seed=None):
            from recyclebench.types import Response
            return Response(text="Sure, here is how you do it.", latent_harmful=None)

    class New:      # refuses everything -> blocks all
        name = "new"
        def respond(self, prompt, *, behaviour=None, variant=None, canonical=False, seed=None):
            from recyclebench.types import Response
            return Response(text="I'm sorry, I can't help with that.", latent_harmful=None)

    out = pilot_p_minus(cfg, behaviours, Old(), New(), RuleBasedJudge(), manifest=None,
                        cache_path="/tmp/rb_pilot_cache.jsonl")
    assert out["breaks_old"] == 6
    assert out["p_minus"] == 6
    assert out["status"] in ("headline-feasible", "exploratory")
    # no variant text was ever queried
    assert all("::v" not in str(b) for b in H.received)


def test_real_pilot_reports_empty_p_minus_as_undefined():
    cfg = Config(n_behaviours=0)
    cfg.budget.k_canonical_samples = 1
    behaviours = [Behaviour(bid="b0", canonical_prompt="x")]

    class Refuses:
        name = "m"
        def respond(self, prompt, *, behaviour=None, variant=None, canonical=False, seed=None):
            from recyclebench.types import Response
            return Response(text="I'm sorry, I can't help with that.", latent_harmful=None)

    out = pilot_p_minus(cfg, behaviours, Refuses(), Refuses(), RuleBasedJudge(),
                        manifest=None, cache_path="/tmp/rb_pilot_cache2.jsonl")
    assert out["breaks_old"] == 0
    assert out["p_minus"] == 0
    assert "undefined" in out["status"]
