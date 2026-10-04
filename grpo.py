#!/usr/bin/env python3
"""Length-penalized GRPO: verl's GRPO trainer with DAPO's soft overlong penalty.

    python grpo.py --model Qwen/Qwen3-8B --revision <rev> --data data --out D [--no-length-penalty]

Each step samples 8 rollouts for each of 16 training prompts (temperature 1.0, top-p 0.95, up to
12,288 new tokens). The reward is +1 for a correct answer and -1 otherwise. The soft overlong
penalty (on by default) adds min(0, -(L - 8,192) / 4,096) for a rollout of L tokens, reaching -1
at the 12,288-token cap; --no-length-penalty gives vanilla GRPO. Advantages are group-normalized;
the clipped policy loss (ratio 0.2, no KL term, no entropy bonus) is averaged over each rollout's
tokens and then over rollouts, in two mini-batches of 8 prompts per step. AdamW at 1e-6 with 10
warmup steps, weight decay 0.1, gradient clipping at 1.0.

Evaluation runs at step 0, every --eval-every steps, and after the last step: 8 samples per
problem at temperature 0.6, top-p 0.95, up to 30,000 new tokens. verl's metrics go to
verl_metrics.jsonl; eval.jsonl holds the per-benchmark accuracy and mean response length.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import data
import opsd

OVERLONG_BUFFER = 4096
OVERLONG_FACTOR = 1.0


def grpo_config(args: argparse.Namespace, model_path: str):
    out = Path(args.out).resolve()
    data_dir = Path(args.data).resolve()
    val_files = "[" + ",".join(f"'{data_dir / f'val_{name}.parquet'}'" for name in args.benchmarks) + "]"
    return opsd.verl_config([
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=true",
        "algorithm.use_kl_in_reward=false",
        "algorithm.kl_ctrl.kl_coef=0.0",
        f"data.train_files={data_dir / 'train.parquet'}",
        f"data.val_files={val_files}",
        f"data.train_batch_size={args.batch_size}",
        "data.val_batch_size=128",
        "data.max_prompt_length=2048",
        f"data.max_response_length={args.max_new_tokens}",
        "data.filter_overlong_prompts=true",
        "data.shuffle=true",
        f"data.seed={args.seed}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.strategy=fsdp",
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.optim.weight_decay=0.1",
        "actor_rollout_ref.actor.grad_clip=1.0",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.clip_ratio=0.2",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.n={args.rollouts_per_prompt}",
        f"actor_rollout_ref.rollout.temperature={opsd.TRAIN_TEMPERATURE}",
        f"actor_rollout_ref.rollout.top_p={opsd.TOP_P}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tp}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
        "actor_rollout_ref.rollout.prompt_length=2048",
        f"actor_rollout_ref.rollout.response_length={args.max_new_tokens}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"+actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed={args.seed}",
        f"actor_rollout_ref.rollout.val_kwargs.n={args.eval_samples}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=true",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={opsd.EVAL_TEMPERATURE}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={opsd.TOP_P}",
        "reward_model.reward_manager=dapo",
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.enable={str(args.length_penalty).lower()}",
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.len={OVERLONG_BUFFER}",
        f"+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor={OVERLONG_FACTOR}",
        "+reward_model.reward_kwargs.overlong_buffer_cfg.log=false",
        f"+reward_model.reward_kwargs.max_resp_len={args.max_new_tokens}",
        f"custom_reward_function.path={opsd.ROOT / 'data.py'}",
        "custom_reward_function.name=compute_score",
        "trainer.critic_warmup=0",
        "trainer.total_epochs=100",
        f"trainer.total_training_steps={args.steps}",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.nnodes=1",
        f"trainer.save_freq={args.steps if args.save_model else -1}",
        f"trainer.test_freq={args.eval_every}",
        f"trainer.val_before_train={str(not args.skip_initial_eval).lower()}",
        "trainer.logger=[console,file]",
        f"trainer.default_local_dir={out / 'checkpoints'}",
        "trainer.resume_mode=disable",
        "trainer.log_val_generations=0",
        f"trainer.validation_data_dir={out / 'eval'}",
        f"+trainer.val_max_response_length={args.eval_max_tokens}",
        "trainer.project_name=hindsight",
        f"trainer.experiment_name={out.name}",
    ])


def write_eval_summary(args: argparse.Namespace, out: Path) -> None:
    """eval.jsonl in the format opsd.py writes, from verl's validation metrics."""
    n = args.eval_samples
    for line in (out / "verl_metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        metrics = record["data"]
        for name in args.benchmarks:
            accuracy = metrics.get(f"val-core/{name}/acc/mean@{n}")
            if accuracy is None:
                continue
            opsd.write_jsonl(out / "eval.jsonl", {
                "step": record["step"], "branch": None, "benchmark": name, "accuracy": accuracy,
                "mean_tokens": metrics.get(f"val-aux/{name}/response_length/mean@{n}"),
                "problems": data.BENCHMARK_ROWS[name], "samples": n})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    opsd.add_common_args(parser, steps=25, batch_size=16)
    parser.set_defaults(max_new_tokens=12288)
    parser.add_argument("--rollouts-per-prompt", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=8, help="Prompts per policy update")
    parser.add_argument("--no-length-penalty", dest="length_penalty", action="store_false",
                        help="Vanilla GRPO: no soft overlong penalty")
    args = parser.parse_args(argv)
    data.load_dotenv_if_present()
    opsd.check_args(args)
    out = Path(args.out)
    if not opsd.claim_output(out, "grpo"):
        return

    from verl.trainer.main_ppo import run_ppo

    model_path = opsd.resolve_model(args.model, args.revision)
    config = grpo_config(args, model_path)
    (out / "config.json").write_text(json.dumps({
        "method": "grpo", **vars(args), "model_path": model_path, "revision": args.revision or opsd.MODELS.get(args.model),
        "overlong_buffer": OVERLONG_BUFFER if args.length_penalty else None, "overlong_penalty_factor": OVERLONG_FACTOR,
        "data_pins": json.loads((Path(args.data) / "manifest.json").read_text())["pins"]}, indent=2) + "\n")
    os.environ["VERL_FILE_LOGGER_PATH"] = str((out / "verl_metrics.jsonl").resolve())
    run_ppo(config)
    write_eval_summary(args, out)
    if args.save_model:
        (out / "model").symlink_to(Path("checkpoints") / f"global_step_{args.steps}" / "actor")
    opsd.mark_done(out)


if __name__ == "__main__":
    main()
