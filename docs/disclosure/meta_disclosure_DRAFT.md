# DRAFT — responsible-disclosure note to Meta (not sent)

> **Status:** draft for the candidate and supervisor. Nothing here has been sent. Before sending: (1) choose the channel
> (confirm Meta's current route for Llama model-behaviour / safety reports with the supervisor; do not rely on a URL from
> this draft); (2) fix the embargo length; (3) regenerate every number from `results/thesis_numbers_*.tex` after the open
> audits (prereg Amendment 2 §B4) are labelled and Amendment 2 is signed; (4) send no prompt or response text in the first
> message — offer it through a controlled channel on request.

---

**Subject:** Research finding: Llama 3.1 8B Instruct refusal fixes over Llama 3 8B Instruct often do not survive
meaning-preserving rewording

Dear Llama safety team,

I am an M.Sc. candidate at RPTU Kaiserslautern-Landau (supervisor: [name]; examiner: Prof. Dr. Sebastian Vollmer), writing a
thesis on whether safety alignment generalises across Llama releases. Ahead of publication I am sharing a finding about
Llama 3.1 8B Instruct, with an embargo of **[N] days** from today, until **[date]**, before the thesis is made public.

**Finding.** Among the SORRY-Bench requests that Llama 3 8B Instruct answered and Llama 3.1 8B Instruct refused (the
behaviours the newer release appears to have fixed), about half were answered again by Llama 3.1 8B Instruct when the request
was reworded without changing what it asks for: **[\rbAnchorResolvedKn] re-opened, [\rbAnchorResolvedRSRcorr] after
correction for judge error [\rbAnchorResolvedRSRcorrCI]**. A registered replication on held-out behaviours gave
[\rbConfKn] ([\rbConfRSRcorr] corrected). The rewordings are plain-language paraphrase, syntactic, lexical, back-translation,
tense and register changes produced by a small non-Llama model (qwen2.5:3b); each one passed an automatic intent check
(embedding similarity, BERTScore, NLI entailment), and a human audit found that 3 of 98 re-opening rewordings had changed the
request. No adversarial optimisation, templates or role-play were used.

**What it suggests.** The fixes behave more like refusals tied to how a request is phrased than to what it asks for. The
newer model does not simply refuse more: on XSTest's benign prompts both releases refuse the same share
([\rbORThreeEight] vs [\rbORThreeOneEight]).

**Severity and scope, stated plainly.**
- The re-opened behaviours skew mild (e.g. insulting, deceptive-review and harassment-adjacent requests in SORRY-Bench's
  taxonomy); this is not a finding about high-severity uplift.
- Model-level only: no Llama Guard, Prompt Guard or other system-level safeguard was in the loop.
- Local Ollama builds, quantised (Llama 3 8B Instruct Q4_0, Llama 3.1 8B Instruct Q4_K_M), temperature 0, Ollama's default
  sampling settings. Behaviour of the full-precision weights may differ.
- Both releases are from 2024; I have not tested Llama 3.3, Llama 4 or the 70B/405B models.

**What I can share.** Aggregate results and the method now. The per-behaviour list, the re-opening rewordings and the model
replies are withheld from the thesis and can be shared with your team through a channel you choose.

**Timeline.** Thesis submission planned for [date]; the public version will report aggregate rates and methodology only, with
no working rewordings or model responses.

Kind regards,
Hatim Mahmoud
[e-mail] · RPTU Kaiserslautern-Landau
