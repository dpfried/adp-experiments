# OpenHands SDK SFT Training With LLaMA-Factory

These scripts reproduce the data extraction, cleaning, and training setup used
for a small full-parameter fine-tuning run on ADP OpenHands-compatible SFT data.

The run documented here used:

- Model: `Qwen/Qwen3.5-0.8B-Base`
- Trainer: LLaMA-Factory SFT
- Training type: full-parameter fine-tuning
- Data mixture: ADP paper-style OpenHands/SWE-Agent non-web mixture
- Sequence length: `2048`
- Per-device batch size: `1`
- Max steps: `10000`
- Eval every: `500` steps
- Eval split size: `290` trajectories
- Hardware observed: AMD ROCm on Radeon 8060S, 32 GiB VRAM allocation from a
  64 GiB unified-memory machine

The local run reached training successfully with 39,956 examples after cleaning
and LLaMA-Factory filtering. It used about 20.2 GiB VRAM and ran at roughly
3.36 seconds per step after ROCm warmup.

## Layout

- `scripts/setup_rocm_uv_env.sh`: create a uv virtual environment and install
  the training dependencies.
- `scripts/download_adp_hf_release.py`: download the full ADP SFT JSONL files
  from Hugging Face.
- `scripts/make_paper_splits.py`: create deterministic paper-style train/eval
  mixtures.
- `scripts/clean_for_llamafactory.py`: remove malformed records and escape
  literal multimodal XML tags that LLaMA-Factory interprets as media markers.
- `scripts/write_llamafactory_config.py`: write `dataset_info.json` entries and
  the Qwen3.5 0.8B training YAML.
- `scripts/run_training.sh`: launch the 10k-step run in the background.
- `scripts/monitor_training.sh`: inspect the running job, logs, and ROCm memory.

## Reproduce

Run from the repository root:

```bash
cd openhands_sdk_training

# Optional but recommended on the machine where training will run.
bash scripts/setup_rocm_uv_env.sh
source .venv/bin/activate

# Authenticate before training if you want W&B logging.
wandb login

# Download the full ADP SFT release. This is large: about 33 GiB for all subsets.
python scripts/download_adp_hf_release.py \
  --repo-id neulab/agent-data-collection \
  --out ~/data/adp_openhands_sdk/hf_release \
  --preset all

# Build the OpenHands/SWE-Agent non-web train/eval split used in this run.
python scripts/make_paper_splits.py \
  --input-root ~/data/adp_openhands_sdk/hf_release \
  --output-dir ~/data/adp_openhands_sdk/balanced_splits \
  --mixture openhands_nonweb

# Validate/clean JSONL and neutralize literal <image>/<video>/<audio> tags.
python scripts/clean_for_llamafactory.py \
  --train ~/data/adp_openhands_sdk/balanced_splits/paper_openhands_nonweb_train.jsonl \
  --eval ~/data/adp_openhands_sdk/balanced_splits/paper_openhands_nonweb_eval.jsonl

# Write LLaMA-Factory dataset metadata and the training YAML.
python scripts/write_llamafactory_config.py \
  --dataset-dir ~/data/adp_openhands_sdk/balanced_splits \
  --output-dir ~/data/adp_openhands_sdk/balanced_splits/output/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe \
  --run-name adp-openhands-nonweb-qwen35-0.8b-full-10k-bs1-seq2048-mm-safe

# Launch in the background.
bash scripts/run_training.sh \
  ~/data/adp_openhands_sdk/balanced_splits/qwen35_0_8b_openhands_nonweb_full_10k_bs1_seq2048_mm_safe.yaml \
  ~/data/adp_openhands_sdk/balanced_splits/logs/train_10k_mm_safe
```

Monitor the run:

```bash
bash scripts/monitor_training.sh ~/data/adp_openhands_sdk/balanced_splits/logs/train_10k_mm_safe
tail -f ~/data/adp_openhands_sdk/balanced_splits/logs/train_10k_mm_safe.log
```

## Data Mixtures

The ADP paper-style weighted mixtures implemented by `make_paper_splits.py` are:

- `openhands_nonweb`: AgentTuning, CodeActInstruct, SWE-Agent/OpenHands, SWE-Gym,
  SWE-Smith, Code Feedback, and Orca AgentInstruct subsets.
- `agentlab_web`: Go-Browse-WA, Mind2Web, NNetNav, and Synatra web subsets.
- `all_weighted_union`: union of the two groups above.

The eval policy used here is a deterministic half-size holdout per source:

```text
previous_eval_count = min(100, max(10, ceil(0.02 * raw_count)))
eval_count = max(1, floor(previous_eval_count / 2))
```

Training rows are sampled from the remaining rows using the ADP appendix
multipliers encoded in the script. Eval rows are never upsampled.

## Notes From The First Run

The initial uncleaned split had two issues:

- 5 malformed rows had null or invalid `conversations` fields and failed
  LLaMA-Factory conversion.
- 14 rows contained literal `<video>`/`<image>`/`<audio>` text. Qwen VL
  processors interpreted these as media placeholders, so the cleaner escapes
  those literal strings before training.

LLaMA-Factory also skipped one abnormal role-pattern example during conversion.
That was non-fatal.

Sequences longer than `cutoff_len: 2048` are truncated by LLaMA-Factory. This
means only the prefix up to the cutoff participates in the loss; labels beyond
that point are dropped for that training example.

