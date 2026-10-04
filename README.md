# On-Policy Self-Distillation (OPSD) with a Hindsight Self-Teacher

Code to train thinking-enabled reasoning models with OPSD and with the comparison
methods: short-trace RFT, length-penalized GRPO, and prompt-matched lockstep OPSD.
In OPSD the student samples a rollout, and a frozen copy of the initial model
(the *self-teacher*) reads the same rollout after a privileged context, such as
the student's own reflection on the ground-truth answer. The student is trained
toward the teacher's next-token distribution at every rollout token.

## Files

| File | Purpose |
|---|---|
| `data.py` | Downloads the pinned training prompts and benchmarks, and generates expert demonstrations |
| `opsd.py` | OPSD training (teacher contexts, gating, losses, reinjection) and lockstep OPSD |
| `rft.py` | Rejection-sampling fine-tuning on the student's own correct rollouts |
| `grpo.py` | GRPO with DAPO's soft overlong penalty |
| `scripts/*.sh` | One end-to-end run per experiment |
| `verl/` | verl v0.7.0 with a few changes (see [Changes to verl](#changes-to-verl)) |
| `requirements.txt` | Python dependencies |

There is no separate evaluation stage: every run evaluates itself at step 0,
every 25 steps, and after its last step.

## Setup

Python 3.10 and four CUDA GPUs with 80 GB each (e.g. H100) are assumed.

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-build-isolation flash-attn==2.8.3
```

`requirements.txt` installs `verl/` in editable mode, so run it from this directory.

Create a `.env` file in this directory (next to `opsd.py`) with two keys:

```bash
OPENAI_API_KEY=sk-...   # generates expert demonstrations (only for the demo-based teacher contexts)
HF_TOKEN=hf_...         # Hugging Face token; the models and datasets are public, but it avoids rate limits
```

Every stage reads `.env` automatically, and variables already exported in your
shell take precedence. `.env` is git-ignored; keep it out of any archive you share.

## Quick start

```bash
bash scripts/qwen3_8b.sh        # seed 42 (default)
bash scripts/qwen3_8b.sh 1      # seed 1
```

Each script prepares the data (once, shared by all runs), then trains and
evaluates. One seed drives the prompt order, sampling, and evaluation. For a
multi-seed result, run the script once per seed and average.

| Script | Model | Method | Steps |
|---|---|---|---|
| `qwen3_8b.sh` | Qwen3-8B | OPSD | 100 |
| `acereason_nemotron_7b.sh` | AceReason-Nemotron-7B | OPSD | 100 |
| `deepseek_r1_distill_qwen_7b.sh` | DeepSeek-R1-Distill-Qwen-7B | OPSD | 100 |
| `qwen3_8b_rft_short.sh` | Qwen3-8B | Short-trace RFT | 25 |
| `qwen3_8b_grpo_length.sh` | Qwen3-8B | Length-penalized GRPO | 25 |
| `qwen3_8b_lockstep.sh` | Qwen3-8B | Prompt-matched lockstep OPSD (correct-only vs. incorrect-only) | 25 |

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CONTEXT` | `reflection_from_gt` | Teacher context (OPSD, lockstep); see [Protocol](#protocol) |
| `GATING` | `all` | Rollouts trained on (OPSD): `all`, `correct`, `incorrect` |
| `LOSS` | `reverse_kl` | OPSD, lockstep: `reverse_kl`, `forward_kl`, `jsd`, `split` (reverse KL on correct rollouts, forward KL on the rest) |
| `REINJECTION` | `0` | `1` repeats the privileged context in the teacher's input every 2,048 tokens of thinking (OPSD) |
| `SELECTION` | `shortest_half` | `all` trains on every correct rollout, i.e. plain RFT (RFT) |
| `LENGTH_PENALTY` | `1` | `0` gives vanilla GRPO (GRPO) |
| `BENCHMARKS` | `math500,aime24,aime25` | Add `aime26,hmmt26,olympiadbench` for the extended suite |
| `GPUS` | `$CUDA_VISIBLE_DEVICES` or `0,1,2,3` | GPUs to use; one run uses all of them |
| `TP` | `2` | Tensor-parallel size of each inference server |
| `GPU_MEMORY_UTILIZATION` | `0.6` | Share of GPU memory for the inference servers |
| `OUT` | `outputs` | Output root |
| `YES` | unset | `YES=1` generates expert demonstrations without asking |
| `DRY_RUN` | unset | `DRY_RUN=1` prints the commands without running them |

Rerunning a script skips a finished run. Training does not checkpoint mid-run,
so an interrupted run starts over; its files are moved to
`<run>.incomplete-<time>`.

The demo-based teacher contexts need expert demonstrations for the 3,200
training prompts. When one of them is selected, the script asks before
generating them with `gpt-5.4-2026-03-05` (answers checked by
`gpt-5.4-nano-2026-03-17`; a rejected solution is regenerated once with the
answer shown). They are written once to `data/expert_demos.parquet`.

The lockstep script swaps the two branches' weights and optimizer state through
disk, which takes about 230 GB under the run directory.

## Outputs

```
data/                              train.parquet, val_<benchmark>.parquet, expert_demos.parquet
outputs/<model>/seed<SEED>/
├── <run>/                         e.g. opsd_reflection_from_gt_all_reverse_kl, rft_shortest_half
│   ├── config.json                settings, model revision, data pins
│   ├── metrics.jsonl              one line per step: rollout accuracy, truncation, loss, reflection stats, timing
│   ├── eval.jsonl                 accuracy and mean response tokens per evaluation step and benchmark
│   ├── eval/[<branch>/]step_<N>/  every evaluation response (GRPO: eval/<step>.jsonl)
│   ├── samples.jsonl              two examples per step: rollout, reflection, teacher input
│   ├── model/huggingface/         trained model (lockstep: model_<branch>/huggingface/)
│   └── done.json
└── logs/<run>.log
```

OPSD runs are named `opsd_<context>_<gating>_<loss>[_reinjection]`. Lockstep
runs also write `matched.jsonl` (the selected prompts and rollouts per step),
and GRPO runs write verl's own metrics to `verl_metrics.jsonl`.

## Running the stages yourself

```bash
python data.py prepare --out data
python data.py demos --out data          # only for the demo-based teacher contexts

python opsd.py --model Qwen/Qwen3-8B --data data --out runs/opsd --seed 42 \
    --context reflection_from_gt --gating correct --loss reverse_kl
python opsd.py --model Qwen/Qwen3-8B --data data --out runs/lockstep --seed 42 --lockstep --steps 25
python rft.py  --model Qwen/Qwen3-8B --data data --out runs/rft --seed 42
python grpo.py --model Qwen/Qwen3-8B --data data --out runs/grpo --seed 42
```

The three models in the scripts default to their pinned revisions; pass
`--revision` for any other Hub model, or a local directory as `--model`. For a
quick smoke test, shrink a run with `--steps 2 --batch-size 8 --eval-samples 1
--benchmarks aime24`. Each Python script lists all of its options with `--help`.

## Protocol

**Data.** Training prompts are the first 3,200 prompts of an 80% split of
DAPO-Math-17k (deduplicated, shuffled with seed 42). Every prompt, in training
and evaluation, asks for the answer on a final `Answer:` line, and answers are
checked by DAPO's rule-based verifier: the last `Answer:` line, else the last
`\boxed{}` (Math-Verify for HMMT26 and OlympiadBench). Training rollouts are
checked over the whole response; evaluation and the GRPO reward check only the
last 300 characters, as verl's DAPO scorer does. Datasets are pinned to Hub
revisions and SHA-256 hashes, and models to Hub revisions.

**OPSD.** Each step samples one rollout for each of 32 prompts (temperature 1.0,
top-p 0.95, up to 8,192 new tokens). Rollouts are used exactly as sampled; one
cut off by the token budget ends without an end-of-sequence token. The
self-teacher is the frozen initial model. The loss is the full-vocabulary
divergence between the student's and the teacher's next-token distributions,
averaged over each rollout's tokens and then over rollouts. AdamW, constant
learning rate 1e-6, gradient clipping at 1.0, 100 steps.

**Teacher contexts.** The teacher's prompt is the student prompt with one of these
privileged contexts added:

| `CONTEXT` | Privileged context |
|---|---|
| `reflection_from_gt` | A memo the student writes after being told whether its rollout was correct and the ground-truth answer |
| `conciseness` | An instruction to solve concisely, in place of the prompt's opening instruction (no reflection) |
| `reflection_from_demo` | As `reflection_from_gt`, with an expert solution shown during the reflection |
| `do_dont` | A two-line DO / DON'T note written against an expert solution |
| `four_line` | A four-part, answer-free reflection written against an expert solution |
| `resolve_from_demo` | A fresh solution the student writes with an expert solution in view |

The reflection is generated by the current student (up to 6,144 new tokens,
continued for up to 2,048 more if cut off), and the teacher reads the text after
its thinking block. A rollout cut off by its token budget is reflected on as
incorrect. With reinjection, the context is repeated inside the
teacher's copy of the rollout ("Actually, I recall: … Continuing from this.")
near every 2,048 tokens of thinking, moved back to the nearest sentence end; the
student's input is unchanged.

**Lockstep.** A correct-only and an incorrect-only branch start from the same
model. Each step streams candidate batches of 32 prompts, and each branch
samples 4 rollouts per prompt. A prompt is kept when the correct-only branch has
a correct rollout and the incorrect-only branch an incorrect one (preferring one
that finished), until 16 prompts are matched. Each branch then reflects on and
trains on its own selected rollouts. 25 steps; other settings as in OPSD.

**RFT.** Each step samples one rollout for each of 64 prompts and keeps the
correct ones; short-trace RFT keeps the shorter half of those by token count.
Cross-entropy on the kept rollouts, averaged as in OPSD. 25 steps; other
settings as in OPSD.

**GRPO.** Each step samples 8 rollouts for each of 16 prompts (temperature 1.0,
top-p 0.95, up to 12,288 new tokens). The reward is +1 for a correct answer and
−1 otherwise, plus DAPO's soft overlong penalty, which falls linearly from 0 at
8,192 tokens to −1 at 12,288. Group-normalized advantages, clip ratio 0.2, no
KL or entropy term, two policy updates of 8 prompts per step, AdamW at 1e-6
with 10 warmup steps and weight decay 0.1. 25 steps.

**Evaluation.** 8 samples per problem at temperature 0.6, top-p 0.95, with up
to 30,000 new tokens, using the training prompt format. Each benchmark reports
mean accuracy (avg@8) and mean response length.

## Changes to verl

`verl/` is a clone of verl v0.7.0 (commit `f9c855f7`). No files are added; the
six below are modified. The OPSD and RFT update and the lockstep branch
snapshots are not part of verl: they live in `opsd.py` as a subclass of verl's
worker.

| File (under `verl/verl/`) | Change | Used by |
|---|---|---|
| `trainer/ppo/ray_trainer.py` | Validation passes `trainer.val_max_response_length` as `max_new_tokens`, records each sample's response length (`val-aux/<benchmark>/response_length/mean@n`), and writes numpy values in its generation dumps as plain JSON | `grpo.py` |
| `experimental/agent_loop/agent_loop.py` | A batch's `max_new_tokens` reaches the sampling parameters, and responses are padded to that length | `grpo.py` |
| `experimental/agent_loop/single_turn_agent_loop.py` | Keeps up to `max_new_tokens` response tokens instead of truncating at the training response length | `grpo.py` |
| `trainer/ppo/metric_utils.py` | Logs `grpo/frac_mixed` (prompts with both correct and incorrect rollouts) and `grpo/mean_group_pass_rate` | `grpo.py` |
| `workers/rollout/utils.py` | Starts the rollout HTTP server with `startup()` and `main_loop()`, so it keeps accepting connections | all |
| `utils/dataset/vision_utils.py` | `qwen_vl_utils` is optional; it is needed only for vision inputs | all |
