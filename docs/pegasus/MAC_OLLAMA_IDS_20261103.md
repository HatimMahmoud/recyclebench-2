# Ollama model identities on the Mac (2026-11-03)

This answers open item 3 of `docs/LLAMA4_SCOUT_STUDY.md` ("no Mac digest is recorded anywhere in the repo").

**How the IDs were read.** They come straight from the Mac's Ollama manifests, with no call to the daemon:
- **ID:** the first 12 hex digits of the sha256 of `~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>`.
  This is what `ollama list` shows.
- **Model layer:** the digest of the GGUF weights, the same blob `tools/mech_probe.py` loads.

Two tags with one ID are the same model.

| tag | ID (`ollama list`) | model layer (GGUF weights) | size |
|---|---|---|---|
| llama2:7b | `78e26419b446` | `sha256:8934d96d3f08982e95922b2b7a2c626a1fe873d7c3b06e8e56d7bc0a1fef9246` | 3.83 GB |
| llama2:13b | `d475bf4c50bc` | `sha256:2609048d349e7c70196401be59bea7eb89a968d4642e409b0e798b34403b96c8` | 7.37 GB |
| llama3:8b | `365c0bd3c000` | `sha256:6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa` | 4.66 GB |
| llama3.1:8b | `46e0c10c039e` | `sha256:667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29` | 4.92 GB |
| llama3.2:1b | `baf6a787fdff` | `sha256:74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45` | 1.32 GB |
| llama3.2:3b | `a80c4f17acd5` | `sha256:dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff` | 2.02 GB |
| qwen2.5:14b (judge, inverter) | `7cdf5a0187d5` | `sha256:2049f5674b1e92b4464e5729975c9689fcfbf0b0e4443ccf10b5339f370f9a54` | 8.99 GB |
| qwen2.5:3b (paraphraser) | `357c53fb659c` | `sha256:5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6` | 1.93 GB |
| gemma4:e4b (intent layer, held-out run only) | `dc35e8d9c606` | `sha256:370c2879f17648d337b0ab4e208fdce348e24d1060064e93d7645daafd56d39c` | 5.49 GB |
| mistral:7b-instruct-v0.2-q4_K_M | `eb14864c7427` | `sha256:a62468bbef4a991a59c574c04f424c0c2241f33b68b5119403b86935b41977ea` | 4.37 GB |
| mistral:7b-instruct-v0.3-q4_K_M | `6577803aa9a0` | `sha256:f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f` | 4.37 GB |

`llama2:latest` has the same ID as `llama2:7b`. `llama3:8b-instruct-q4_K_M` (`9b8f3f3385bf`) and `qwen2.5:latest`
(`845dbda0ea48`) are on the Mac too, but no study configuration uses them.

## Against Pegasus

`docs/LLAMA4_SCOUT_STUDY.md` gives the Pegasus IDs of four of these models, and all four match the Mac:
- llama3:8b `365c0bd3c000`
- llama3.2:3b `a80c4f17acd5`
- qwen2.5:14b `7cdf5a0187d5`
- qwen2.5:3b `357c53fb659c`

So the Scout recycle and RPE runs used the same weights as the Mac for those models.

**Still to compare:** llama2:7b, llama2:13b, llama3.1:8b and llama3.2:1b. Every Pegasus job prints `ollama list` into
`$RB_LOGS/provenance_<job>.txt` (`pegasus/in_container.sh`), so one look at any job's provenance file closes it. A
model whose ID differs is a different blob. Its Pegasus replies are then a separate environment from the Mac's cache,
and every run that mixes the two must say so.
