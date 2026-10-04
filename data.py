#!/usr/bin/env python3
"""Training prompts, evaluation sets, answer checking, and expert demonstrations.

    python data.py prepare --out data          # training prompts and evaluation sets
    python data.py demos --out data [--yes]    # expert demonstrations (OpenAI API; asks first)

Every prompt, for training and for evaluation, is the DAPO-Math prompt:

    Solve the following math problem step by step. The last line of your response should be of
    the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

    {problem}

    Remember to put your answer on its own line after "Answer:".

Training uses the first 3,200 prompts of an 80% split of DAPO-Math-17k (deduplicated, shuffled
with seed 42). Every source is pinned to a Hub revision and checked against its SHA-256.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

STUDENT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be "
    "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
REMEMBER = '\n\nRemember to put your answer on its own line after "Answer:".'

TRAIN_ROWS = 3200
TRAIN_SPLIT_SEED = 42
TRAIN_FRACTION = 0.8

# repo, revision, file, sha256
DAPO = ("YouJiacheng/DAPO-Math-17k-dedup", "6e26a33abdabd3e6aaa1d742326b790758f7dbc5",
        "distinct-prompts-with-rewards.parquet",
        "c65f47247b9b13cb62321462243c2d1fe8d4ede2d800644a7dda61fea76d4286")
BENCHMARKS = {
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "test.jsonl",
                "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"),
    "aime24": ("math-ai/aime24", "83a7f387baaa524a8bda0022eac0541582297103", "test-00000-of-00001.parquet",
               "2602156eb2a0ab64393dc46617a3bfee3451ae3c985a7ea6ac2b55436e094a38"),
    "aime25": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "test.jsonl",
               "b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331"),
    "aime26": ("math-ai/aime26", "79037aebdb6580008fb960d17cb21fd3099083e3", "aime2026.jsonl",
               "52822957957a3f577d1e9706c36a66a8108a3f99b6aff424cfb72dff0094a9ee"),
    "hmmt26": ("MathArena/hmmt_feb_2026", "02fba4f74d8e68e73e66a02d540fd979c05c274c",
               "data/train-00000-of-00001.parquet",
               "e5fcff6b1c2262841c0c37bf6d7b42529f284528d5f0d5c45c52e8bf0654a916"),
    "olympiadbench": ("Hothan/OlympiadBench", "91184b52131e7fc9455fef848035173aea8cc01a",
                      "OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet",
                      "7c17ca2662b3ea77cddcb66a64b04ad9b53d3d32fa372d444a4884739eb1638d"),
}
DEFAULT_BENCHMARKS = ("math500", "aime24", "aime25")
BENCHMARK_ROWS = {"math500": 500, "aime24": 30, "aime25": 30, "aime26": 30, "hmmt26": 33, "olympiadbench": 674}
SYMBOLIC = {"hmmt26", "olympiadbench"}  # answers checked with Math-Verify
SCORED_TAIL = 300  # verl's DAPO scorer, used for evaluation and the GRPO reward, reads the last 300 characters

DEMO_MODEL = "gpt-5.4-2026-03-05"
DEMO_JUDGE = "gpt-5.4-nano-2026-03-17"
DEMO_MAX_TOKENS = 2048


def student_prompt(problem: str) -> str:
    return STUDENT_PREFIX + problem + REMEMBER


# ---------------------------------------------------------------------------
# Answer checking (training verification, GRPO reward, and evaluation)
# ---------------------------------------------------------------------------


def _symbolic_candidates(ground_truth: str, extra_info: dict | None) -> list[str]:
    answers = (extra_info or {}).get("answers")  # a numpy array when read back from parquet
    answers = [str(a) for a in (list(answers) if answers is not None else []) if str(a).strip()]
    if not answers:
        return [str(ground_truth)]
    if (extra_info or {}).get("multiple_answers") and len(answers) > 1:
        answers.insert(0, "(" + ",".join(answers) + ")")
    return answers


def check_answer(text: str, ground_truth: str, data_source: str = "math_dapo",
                 extra_info: dict | None = None, tail: int | None = None) -> tuple[bool, str]:
    """Return (correct, normalized prediction) for a response.

    The prediction is the last "Answer:" line; if there is none, the last \\boxed{}. With `tail`,
    both are looked for only in the last `tail` characters (SCORED_TAIL for evaluation and the GRPO
    reward, as verl's DAPO scorer does); training rollouts are checked over the whole response.
    HMMT26 and OlympiadBench answers can be symbolic and are checked with Math-Verify on the whole
    response.
    """
    from verl.utils.reward_score.math_dapo import (
        last_boxed_only_string, normalize_final_answer, remove_boxed, verify,
    )

    scored = text[-tail:] if tail else text
    correct, pred = verify(scored, str(ground_truth))
    if pred == "[INVALID]":
        boxed = last_boxed_only_string(scored)
        if boxed is not None:
            try:
                pred = normalize_final_answer(remove_boxed(boxed))
            except (AssertionError, ValueError):
                pred = "[INVALID]"
            correct = pred == normalize_final_answer(str(ground_truth))
    if data_source in SYMBOLIC:
        from verl.utils.reward_score.math_verify import compute_score as math_verify

        correct = False
        for gold in _symbolic_candidates(ground_truth, extra_info):
            try:
                if math_verify(text, gold):
                    correct = True
                    break
            except Exception:
                continue
    return bool(correct), pred


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None, **kwargs: Any) -> dict:
    """verl reward-function interface: +1 for a correct answer, -1 otherwise."""
    correct, pred = check_answer(solution_str, ground_truth, data_source, extra_info, tail=SCORED_TAIL)
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred}


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(repo: str, revision: str, filename: str, expected: str) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{repo}@{revision}/{filename}: sha256 {actual} != pinned {expected}")
    return path


def read_records(path: Path) -> list[dict]:
    import pandas as pd

    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rl_row(data_source: str, index: int, problem: str, answer: str, **extra: Any) -> dict:
    """One row in verl's RL dataset format."""
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": student_prompt(problem)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"index": index, "problem": problem, "ground_truth": str(answer), **extra},
    }


def training_rows() -> list[dict]:
    """The first 3,200 prompts of the 80% DAPO-Math split used by every method."""
    import pandas as pd

    frame = pd.read_parquet(fetch(*DAPO))
    frame = frame.sample(frac=1.0, random_state=TRAIN_SPLIT_SEED).reset_index(drop=True)
    frame = frame.iloc[: int(len(frame) * TRAIN_FRACTION)].iloc[:TRAIN_ROWS]
    rows = []
    for index, record in enumerate(frame.to_dict("records")):
        content = record["prompt"][0]["content"]
        if not (content.startswith(STUDENT_PREFIX) and content.endswith(REMEMBER)):
            raise RuntimeError(f"DAPO prompt {index} does not follow the expected template")
        problem = content[len(STUDENT_PREFIX): -len(REMEMBER)]
        assert student_prompt(problem) == content
        rows.append(rl_row("math_dapo", index, problem, record["reward_model"]["ground_truth"]))
    return rows


def benchmark_rows(name: str) -> list[dict]:
    from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed

    rows = []
    for index, record in enumerate(read_records(fetch(*BENCHMARKS[name]))):
        if name == "aime24":
            rows.append(rl_row(name, index, record["problem"], remove_boxed(last_boxed_only_string(record["solution"]))))
        elif name in ("math500", "aime25", "aime26"):
            rows.append(rl_row(name, index, record["problem"], record["answer"]))
        elif name == "hmmt26":
            problem = str(record["problem"]).removeprefix("Problem:\n").split("\n\nPlease reason step by step")[0]
            rows.append(rl_row(name, index, problem.strip(), record["answer"]))
        else:  # olympiadbench: English, text-only, open-ended maths
            answers = record["final_answer"]
            answers = [str(a) for a in (answers.tolist() if hasattr(answers, "tolist") else answers)]
            problem = str(record["question"])
            if record.get("context"):
                problem = f"{record['context']}\n\n{problem}"
            if "<img_" in problem:
                raise RuntimeError(f"OlympiadBench row {index} is not text-only")
            rows.append(rl_row(name, index, problem, answers[0], answers=answers,
                               multiple_answers=bool(record.get("is_multiple_answer", False))))
    if len(rows) != BENCHMARK_ROWS[name]:
        raise RuntimeError(f"{name} has {len(rows)} rows; expected {BENCHMARK_ROWS[name]}")
    return rows


def write_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(path)


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    manifest_path = out / "manifest.json"
    manifest = {"train": DAPO, "benchmarks": BENCHMARKS, "train_rows": TRAIN_ROWS,
                "student_prompt": student_prompt("{problem}")}
    if manifest_path.exists() and json.loads(manifest_path.read_text())["pins"] == json.loads(json.dumps(manifest)):
        print(f"[data] {out} is up to date", flush=True)
        return
    write_parquet(training_rows(), out / "train.parquet")
    for name in BENCHMARKS:
        write_parquet(benchmark_rows(name), out / f"val_{name}.parquet")
    counts = {name: BENCHMARK_ROWS[name] for name in BENCHMARKS}
    manifest_path.write_text(json.dumps({"pins": manifest, "train": TRAIN_ROWS, "benchmarks": counts}, indent=2) + "\n")
    print(f"[data] wrote {out}: {TRAIN_ROWS} training prompts; benchmarks {counts}", flush=True)


# ---------------------------------------------------------------------------
# Expert demonstrations (reflection from demo, do and don't, four-line reflection, re-solve)
# ---------------------------------------------------------------------------


def demo_answer(text: str) -> str:
    for line in reversed(text.strip().split("\n")):
        if line.strip().lower().startswith("answer:"):
            return line.strip()[len("answer:"):].strip().strip("$").strip()
    return ""


async def demo_for(client, row: dict, semaphore: asyncio.Semaphore) -> dict:
    """One expert solution; if the judge rejects it, one retry that is told the answer."""
    prompt, truth = row["prompt"][0]["content"], row["extra_info"]["ground_truth"]

    async def ask(model: str, messages: list[dict], max_tokens: int) -> str:
        async with semaphore:
            for attempt in range(5):
                try:
                    reply = await client.chat.completions.create(
                        model=model, messages=messages, max_completion_tokens=max_tokens)
                    return reply.choices[0].message.content or ""
                except Exception as exc:  # rate limits and transient API errors
                    if attempt == 4:
                        raise
                    print(f"[demos] retrying after: {exc}", file=sys.stderr, flush=True)
                    await asyncio.sleep(min(60, 2 ** attempt))

    solution = await ask(DEMO_MODEL, [{"role": "user", "content": prompt}], DEMO_MAX_TOKENS)
    answer = demo_answer(solution)
    verdict = ""
    if answer:
        verdict = await ask(DEMO_JUDGE, [
            {"role": "system", "content": "You are a math answer checker. You will be given a ground truth "
             "answer and a candidate answer extracted from a solution. Determine if they are mathematically "
             "equivalent.\n\nRespond with exactly one word: YES or NO."},
            {"role": "user", "content": f"Ground truth answer: {truth}\nCandidate answer: {answer}\n\n"
             "Are these mathematically equivalent? Reply YES or NO."},
        ], 16)
    gt_assisted = not verdict.strip().upper().startswith("YES")
    if gt_assisted:
        solution = await ask(DEMO_MODEL, [{"role": "user", "content": f"{prompt}\n\nThe correct answer is: "
                                            f"{truth}\nSolve the problem and explain why this is the answer."}],
                             DEMO_MAX_TOKENS)
        answer = demo_answer(solution)
    return {"index": row["extra_info"]["index"], "problem": row["extra_info"]["problem"],
            "expert_demonstration": solution, "extracted_answer": answer, "gt_assisted": gt_assisted}


async def generate_demos(rows: list[dict], concurrency: int) -> list[dict]:
    from openai import AsyncOpenAI

    client, semaphore = AsyncOpenAI(), asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(demo_for(client, row, semaphore)) for row in rows]
    started, results = time.time(), []
    for done, task in enumerate(asyncio.as_completed(tasks), start=1):
        results.append(await task)
        if done % 100 == 0 or done == len(tasks):
            print(f"[demos] {done}/{len(tasks)} ({time.time() - started:.0f}s)", flush=True)
    return sorted(results, key=lambda r: r["index"])


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


def cmd_demos(args: argparse.Namespace) -> None:
    import pandas as pd

    out = Path(args.out)
    path = out / "expert_demos.parquet"
    if path.exists() and len(pd.read_parquet(path)) == TRAIN_ROWS:
        print(f"[demos] {path} already holds {TRAIN_ROWS} demonstrations", flush=True)
        return
    rows = pd.read_parquet(out / "train.parquet").to_dict("records")
    print(
        "Expert demonstrations are needed for this teacher context.\n"
        f"  {len(rows)} training problems will be sent to {DEMO_MODEL} (up to {DEMO_MAX_TOKENS} tokens each).\n"
        f"  Each answer is checked by {DEMO_JUDGE}; a rejected solution is regenerated once with the\n"
        "  correct answer shown to the model.\n"
        f"  This uses your OPENAI_API_KEY and writes {path}.",
        flush=True,
    )
    if not args.yes:
        if not sys.stdin.isatty():
            sys.exit("[demos] Not generated: no terminal to confirm. Rerun interactively, or pass --yes.")
        if input("Generate the expert demonstrations now? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("[demos] Not generated.")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[demos] OPENAI_API_KEY is not set (environment or .env).")
    demos = asyncio.run(generate_demos(rows, args.concurrency))
    write_parquet(demos, path)
    print(f"[demos] wrote {path}: {sum(not d['gt_assisted'] for d in demos)} solved unaided, "
          f"{sum(d['gt_assisted'] for d in demos)} answer-assisted", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Training prompts and evaluation sets")
    prepare.add_argument("--out", default=str(ROOT / "data"))
    demos = commands.add_parser("demos", help="Expert demonstrations for the demo-based teacher contexts")
    demos.add_argument("--out", default=str(ROOT / "data"))
    demos.add_argument("--yes", action="store_true", help="Generate without asking")
    demos.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    load_dotenv_if_present()
    cmd_prepare(args) if args.command == "prepare" else cmd_demos(args)


if __name__ == "__main__":
    main()
