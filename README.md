# ADP Experiments

This repository contains small, reproducible experiment scripts around the
Agent Data Protocol (ADP) data release.

Current contents:

- `openhands_sdk_training/`: reproduce the Qwen3.5 0.8B LLaMA-Factory run on
  the ADP OpenHands/SWE-Agent non-web mixture.

## Local Workspace Convention

Keep source code and generated experiment artifacts separate:

- Code checkouts and experiment scripts: `~/work/adp/`
- Data, caches, logs, model outputs, and generated artifacts: `~/exp/adp/`

Recommended experiment layout:

```text
~/work/adp/
├── adp-experiments/
└── agent-data-protocol/

~/exp/adp/
├── datasets/
├── runs/
├── cache/
└── tmp/
```

