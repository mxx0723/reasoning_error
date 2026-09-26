# Clinical judgment updating with Qwen

MedReason-Dx clinical judgment screening for UPDATE and MAINTAIN cases, followed by controlled prior-label behavioral experiments.

The current implementation is [standard_v2](qwen_standard_v2/README.md). It covers diagnosis, investigation selection, and treatment/management questions.

**DeepSeek V4 Flash open weights:** see [DEEPSEEK_V4_FLASH.md](DEEPSEEK_V4_FLASH.md). The default portable profile uses **offline vLLM in the same Python process**, with the official V4 message encoder and local weights. No HTTP inference API, paid provider or API key is used. R1 Distill / Qwen also have an optional [Transformers adapter](DEEPSEEK_LOCAL.md).

## Pipeline

1. Original full-question correctness gate: at least 2/3 independent answers.
2. Source-grounded evidence partitioning into E1 and K.
3. Development screening and independent pair stability checks.
4. UPDATE: minimal evidence search within prespecified evidence groups. MAINTAIN: retain a relevant nondecisive evidence set.
5. Independent final holdout before Fresh / Sequential / fixed-prior comparisons.
6. Semantic and clinical review before formal eligibility.

Behaviorally validated candidates are not automatically validated reasoning errors. This repository does not claim that a steering vector or effective probe has been trained.

## Tests

The protocol tests require only Python's standard library:

```sh
python -m unittest discover -s qwen_standard_v2 -p 'test_*.py' -v
```

## Cluster deployment

`runner.py` uses PyTorch and Transformers with a locally stored Qwen checkpoint and two GPUs. The tested cluster environment used Python 3.9, PyTorch 2.6 and Transformers 4.48.1.

`prepare.py` is currently a migration tool for the existing Skynet experiment directory layout, not a standalone dataset downloader. It expects the original `oeq.jsonl`, prior protocol/model configuration, and historical manual/atomic construction files in sibling experiment directories. Adjust the deployment paths and scheduler settings before use elsewhere.

Preparation produces `sources.json`, `seed_plans.json`, `reviews.json`, `protocol.json`, coverage/migration manifests and a Slurm launcher. GPU jobs must be explicitly submitted for selected source indices. Preparation and tests do not submit jobs.

Dataset contents, model weights, credentials, generated runtime configuration and experiment outputs are not included. Historical pass counts have not been certified under this new protocol. See the Chinese [protocol documentation](qwen_standard_v2/README.md) for thresholds, review format and resume behavior.
