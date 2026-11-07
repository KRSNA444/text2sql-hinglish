"""Merge the LoRA adapter into the base model and save as 16-bit safetensors (for Ollama import).

Run (from C:\\text2sql, inside the Unsloth venv):
    python export_merged.py
"""
import argparse

from unsloth import FastLanguageModel

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="outputs/lora_sql_hinglish")
ap.add_argument("--out", default="outputs/merged_hinglish")
args = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=1536, load_in_4bit=True)
model.save_pretrained_merged(args.out, tok, save_method="merged_16bit")
print("saved merged model to", args.out)
