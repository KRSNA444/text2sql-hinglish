"""Upload the LoRA adapter + a model card to Hugging Face.

Run (from C:\\text2sql, inside the Unsloth venv), after setting the token in this PowerShell window:
    $env:HF_TOKEN="hf_xxx"
    python upload_hf.py
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="outputs/lora_sql_hinglish")
ap.add_argument("--repo", default="Krsna444/text2sql-hinglish-lora")
args = ap.parse_args()

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit('HF_TOKEN not set. In PowerShell: $env:HF_TOKEN="hf_..."')

CARD = """---
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - text-to-sql
  - qlora
  - unsloth
  - hinglish
datasets:
  - xlangai/spider
---

# Text-to-SQL LoRA adapter (English + Hinglish)

LoRA adapter for `Qwen/Qwen2.5-Coder-3B-Instruct`, fine-tuned with QLoRA (Unsloth, rank 16, 1 epoch) on Spider train
plus 600 Hinglish-translated questions. Trained on a single RTX 4060 8 GB laptop GPU.

Code, evaluation scripts and demo app: https://github.com/KRSNA444/text2sql-hinglish

## Results (execution accuracy, same script and settings for all rows)

| Model | English (Spider dev, 1,034 q) | Hinglish (200 q) |
|---|---|---|
| Base Qwen2.5-Coder-3B-Instruct | 62.3% | 46.5% |
| Fine-tuned, English only | 72.2% | 55.5% |
| **This adapter (English + 600 Hinglish)** | 71.6% | 66.0% |

## Prompt format

The model expects a system prompt, then a user message of the form:

```
Database schema:
CREATE TABLE singer (
  singer_id number,
  name text,
  PRIMARY KEY (singer_id)
);

Question: Kitne singers hain?

SQL:
```

See `common.py` in the GitHub repo for the exact system prompt and schema formatter.

## Limitations

- Single run, single seed. Hinglish test questions were produced the same way as the Hinglish training questions
  (LLM-assisted translation), so the Hinglish number may be optimistic for real users.
- Trained and evaluated on Spider (small, clean schemas). Not tested on large real-world databases.
- Always run generated SQL read-only; the demo app allows only a single SELECT.

## Licenses and credits

Dataset: Spider (Yu et al., 2018), CC BY-SA 4.0. Base model by Alibaba Qwen: check the base model's license
before commercial use.
"""

api = HfApi(token=token)
api.create_repo(args.repo, repo_type="model", exist_ok=True)
Path(args.adapter, "README.md").write_text(CARD, encoding="utf-8")
api.upload_folder(
    folder_path=args.adapter, repo_id=args.repo, repo_type="model",
    ignore_patterns=["checkpoint-*", "*.log", "train_log.json"],
)
print("Done: https://huggingface.co/" + args.repo)
