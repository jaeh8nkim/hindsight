#!/usr/bin/env python3
"""Rejection-sampling fine-tuning (RFT) on the student's own correct rollouts.

    python rft.py --model Qwen/Qwen3-8B --revision <rev> --data data --out D [--selection all]

Each step samples one rollout per training prompt from the current student (64 prompts,
temperature 1.0, top-p 0.95, up to 8,192 new tokens) and keeps the correct ones. With the default
--selection shortest_half, only the shorter half of the correct rollouts (by token count, at least
one) is kept: short-trace RFT. The student is trained with cross-entropy on the kept rollouts'
tokens, averaged over each rollout's tokens and then over rollouts. A step without a correct
rollout makes no update.

Training and evaluation otherwise match opsd.py: AdamW at a constant learning rate of 1e-6,
gradient clipping at 1.0, and evaluation at step 0, every --eval-every steps, and after the last
step (8 samples per problem at temperature 0.6, top-p 0.95, up to 30,000 new tokens).
"""

import argparse

import opsd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    opsd.add_common_args(parser, steps=25, batch_size=64)
    parser.add_argument("--selection", choices=("shortest_half", "all"), default="shortest_half",
                        help="Which correct rollouts to train on")
    opsd.run(parser.parse_args(argv), "rft")


if __name__ == "__main__":
    main()
