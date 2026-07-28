# Benchmark data (not shipped)

No harmful prompts ship with recyclebench. To run against real models, place your own
benchmark CSV here with a `Goal` column (JailbreakBench / AdvBench format), e.g.:

```
Goal,Category
<prompt text>,privacy
```

Recommended sources (obtain and comply with their own terms):
JBB-Behaviors, HarmBench, StrongREJECT, SORRY-Bench, AdvBench, in-the-wild collections.

Pool several benchmarks: the binding constraint on the primary metric is |P^-|, so a larger
and more diverse behaviour pool directly improves the study's power.
