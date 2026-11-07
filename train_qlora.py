"""QLoRA fine-tuning of Qwen2.5-Coder-3B-Instruct for Text-to-SQL with Unsloth.

Run with the Unsloth Studio environment's Python (from C:\\text2sql):

  # 1) quick smoke test: ~20 steps, tells you speed + GPU memory
  & "C:\\Users\\GIGABYTE\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe" train_qlora.py --max-steps 20

  # 2) real run (1 epoch first)
  & "C:\\Users\\GIGABYTE\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe" train_qlora.py --epochs 1

Input : data/train.jsonl, data/val.jsonl   (made by prepare_data.py)
Output: outputs/lora_<tag>/   (LoRA adapter) + outputs/lora_<tag>/train_log.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="outputs/lora_sql")
    ap.add_argument("--max-seq-length", type=int, default=1536)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="set e.g. 20 for a smoke test")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    # unsloth must be imported before transformers / trl / peft
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto (bf16 on RTX 40-series)
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # big VRAM saver
        random_state=args.seed,
    )

    def to_text(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    def load_split(path):
        ds = load_dataset("json", data_files=path, split="train")
        ds = ds.map(to_text, batched=True, num_proc=1)
        # drop examples that would be truncated (truncation would cut off the SQL answer)
        n0 = len(ds)
        ds = ds.filter(
            lambda ex: len(tokenizer(ex["text"], add_special_tokens=False)["input_ids"])
            <= args.max_seq_length
        )
        print(f"{path}: kept {len(ds)} / {n0} examples (max_seq_length={args.max_seq_length})")
        return ds

    train_ds = load_split(args.train)
    val_ds = load_split(args.val)

    cfg = SFTConfig(
        output_dir=str(Path(args.out) / "checkpoints"),
        dataset_text_field="text",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps" if args.max_steps < 0 else "no",
        eval_steps=100,
        save_strategy="steps" if args.max_steps < 0 else "no",
        save_steps=100,
        save_total_limit=2,
        seed=args.seed,
        report_to="none",
        dataset_num_proc=1,      # Windows: avoid multiprocessing problems
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=cfg,
    )

    # compute loss only on the assistant answer (the SQL), not on schema/question
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    torch.cuda.reset_peak_memory_stats()
    stats = trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"\nTraining time: {stats.metrics['train_runtime']:.0f} s")
    print(f"Peak GPU memory: {peak_gb:.2f} GB")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "train_log.json").write_text(json.dumps({
        "args": vars(args),
        "metrics": stats.metrics,
        "peak_gpu_gb": round(peak_gb, 2),
        "log_history": trainer.state.log_history,
    }, indent=2))
    print(f"Adapter saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
