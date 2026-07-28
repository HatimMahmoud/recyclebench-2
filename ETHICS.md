# Ethics, dual-use, and responsible disclosure

This is **defensive security research**, mirroring the exposé's Sec. 10.

## No new harm surface
All harmful behaviours are reused verbatim from public benchmarks that the user supplies;
none is created here. **No harmful content ships with this repository.** The synthetic
behaviours are benign placeholders (`[benign-placeholder N]`) carrying only latent numeric
parameters used to validate the estimators. The reformulation templates are generic
rewrite instructions containing no harmful exemplars.

## Containment
Experiments are designed to run offline against locally hosted (or privately rented,
researcher-controlled) open-weight models. No harmful output is deployed, served, or
distributed.

## Minimal-disclosure reporting
The released artefact reports aggregate statistics, metrics, and methodology. Concrete
working reworded prompts and verbatim harmful completions are withheld, and shared -- if at
all -- only through controlled academic channels. `store_raw` defaults to `false`, so raw
completions are not persisted; enabling it is a deliberate act with disclosure consequences.

## The annotation sheets
Human annotation of intent preservation requires annotators to read the original request and
its rewrite, so `audit-export` writes prompt text to disk. That sheet is a **working file,
not a released artefact**: it is written outside `results/`, is git-ignored by default, and
must not be published with the instrument. The judge-calibration sheet is stricter still: it
contains model *responses* to harmful requests. Both are written outside `results/`, are
git-ignored, and only the aggregate rates, calibration figures, and the threshold
sensitivity table are reportable.

## The response cache
`results/cache.jsonl` stores model completions to the supplied prompts so that an expensive
run is resumable and auditable. It therefore contains model output to harmful requests: it is
git-ignored, must not be released, and can be disabled entirely (`use_cache=False`) where
even local persistence is inappropriate.

## Responsible disclosure
Actionable model-specific weaknesses are reported to the provider before publication, with
an embargo window.

## Oversight
The project follows the supervising group's research-ethics guidance and institutional
review where applicable.

## A note on the mock backend
The mock path exists to validate the *instrument*, not to make claims about any model. Its
numbers are synthetic by construction and must never be presented as findings about Llama or
any other system.
