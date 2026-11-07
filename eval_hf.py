"""Execution-accuracy evaluation using Unsloth / Hugging Face directly (no Ollama).

Runs the base model and the fine-tuned model in the SAME setup, so the before/after
comparison is fair.

Usage (from C:\\text2sql, with the Unsloth Studio python):
  # base model (before)
  python eval_hf.py --out results/hf_base.jsonl
  # fine-tuned model (after)
  python eval_hf.py --adapter outputs/lora_sql --out results/hf_finetuned.jsonl
  # quick test
  python eval_hf.py --adapter outputs/lora_sql --limit 50 --out results/hf_test.jsonl
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--adapter", default="", help="LoRA folder; empty = evaluate the base model")
    ap.add_argument("--data", default="data/dev.jsonl")
    ap.add_argument("--db-dir", default="spider/database")
    ap.add_argument("--out", default="results/hf_eval.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    # unsloth first, then the rest
    from unsloth import FastLanguageModel
    import torch
    from common import SYSTEM_PROMPT, build_user_prompt, extract_sql
    from eval_exec import run_sql, same_result

    name = args.adapter if args.adapter else args.base
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": build_user_prompt(r["schema"], r["question"])}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in rows
    ]

    # generate in length-sorted batches (less padding), keep original order for output
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    raw = [None] * len(rows)
    t0 = time.time()
    for b in range(0, len(order), args.batch_size):
        idxs = order[b: b + args.batch_size]
        enc = tokenizer([prompts[i] for i in idxs], return_tensors="pt", padding=True,
                        add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for k, i in enumerate(idxs):
            raw[i] = tokenizer.decode(out[k][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        done = min(b + args.batch_size, len(order))
        if (b // args.batch_size) % 5 == 0 or done == len(order):
            print(f"generated {done}/{len(order)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_valid = n_correct = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, (ex, text) in enumerate(zip(rows, raw), 1):
            db_path = Path(args.db_dir) / ex["db_id"] / f"{ex['db_id']}.sqlite"
            pred = extract_sql(text)
            valid = correct = False
            error = ""
            try:
                pred_rows = run_sql(db_path, pred)
                valid = True
                gold_rows = run_sql(db_path, ex["gold_sql"])
                ordered = "order by" in ex["gold_sql"].lower()
                correct = same_result(gold_rows, pred_rows, ordered)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            n_valid += valid
            n_correct += correct
            fout.write(json.dumps({
                "idx": i, "db_id": ex["db_id"], "question": ex["question"],
                "gold_sql": ex["gold_sql"], "pred_sql": pred,
                "valid": valid, "correct": correct, "error": error,
            }, ensure_ascii=False) + "\n")

    n = len(rows)
    summary = {
        "model": name, "adapter": bool(args.adapter), "data": args.data, "examples": n,
        "valid_sql_rate": round(n_valid / n, 4),
        "execution_accuracy": round(n_correct / n, 4),
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
