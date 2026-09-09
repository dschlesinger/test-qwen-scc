#!/usr/bin/env python3
"""Tokens/sec benchmark for Qwen models on a single GPU (SCC)."""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPT = (
    "Explain the difference between supervised and unsupervised learning "
    "in three paragraphs, with a concrete example for each."
)


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return tokenizer, model


def run_once(tokenizer, model, prompt: str, max_new_tokens: int):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    input_len = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    new_tokens = output.shape[1] - input_len
    elapsed = t1 - t0
    return {
        "input_tokens": input_len,
        "output_tokens": new_tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": new_tokens / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--output", default="benchmark_result.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. Check qsub GPU request / drivers.")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"Model: {args.model}")

    tokenizer, model = load_model(args.model)

    print(f"Warmup: {args.warmup_runs} run(s)")
    for _ in range(args.warmup_runs):
        run_once(tokenizer, model, args.prompt, args.max_new_tokens)

    print(f"Timed: {args.timed_runs} run(s)")
    results = []
    for i in range(args.timed_runs):
        r = run_once(tokenizer, model, args.prompt, args.max_new_tokens)
        print(
            f"  run {i + 1}: {r['output_tokens']} tok in {r['elapsed_s']:.2f}s "
            f"-> {r['tokens_per_sec']:.2f} tok/s"
        )
        results.append(r)

    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    summary = {
        "model": args.model,
        "gpu": gpu_name,
        "gpu_mem_gb": gpu_mem_gb,
        "max_new_tokens": args.max_new_tokens,
        "runs": results,
        "avg_tokens_per_sec": avg_tps,
    }

    print(f"\nAverage: {avg_tps:.2f} tokens/sec")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
