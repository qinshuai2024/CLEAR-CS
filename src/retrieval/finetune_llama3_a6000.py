import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

# Configuration
# Update these paths according to your setup:

# For QReCC:
# model_name = "models/meta-llama/Llama-3.1-8B"
# data_file = "data/qrecc/LLM_retrieval/finetuning_data.jsonl"
# output_dir = "experiments/LLM_retrieval/llama31_8b_finetuned_qrecc"
# logging_dir = "experiments/LLM_retrieval/logs"
# qa_adapter = "experiments/LLM_retrieval/llama3_qa_adapter_qrecc"

# For TopiOCQA:
# model_name = "models/meta-llama/Llama-3.1-8B"
# data_file = "data/topiocqa/LLM_retrieval/finetuning_data.jsonl"
# output_dir = "experiments/LLM_retrieval/llama31_8b_finetuned_topiocqa"
# logging_dir = "experiments/LLM_retrieval/logs"
# qa_adapter = "experiments/LLM_retrieval/llama3_qa_adapter_topiocqa"

# Default configuration (update these paths)
model_name = "models/meta-llama/Llama-3.1-8B"
data_file = "data/finetuning_data.jsonl"
output_dir = "experiments/LLM_retrieval/llama31_8b_finetuned"
logging_dir = "experiments/LLM_retrieval/logs"
qa_adapter = "experiments/LLM_retrieval/llama3_qa_adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
full_dataset = load_dataset("json", data_files=data_file, split="train", field=None)

# Split train/eval (e.g., 90/10)
split_datasets = full_dataset.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": split_datasets["train"],
    "eval": split_datasets["test"]
})

# Format dialogue
def format_example(example):
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"{role}: {content}\n"
    return {"text": text}

dataset = dataset.map(format_example)

# Tokenize and labels
def tokenize(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, batched=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training configuration (with WandB)
wandb.init(project="lora-llama-qrecc", mode="offline") 

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    bf16=True,
    eval_strategy="steps",   # Evaluate every eval_steps
    eval_steps=100,
    logging_dir=logging_dir,
    report_to=["tensorboard", "wandb"],  # TensorBoard + W&B
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save LoRA weights
model.save_pretrained(qa_adapter)