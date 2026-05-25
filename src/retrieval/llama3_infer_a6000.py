import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import os

# ==========================
# Configuration
# ==========================
# Example configurations (update paths according to your setup):

# For QReCC:
# BASE_MODEL_PATH = "models/meta-llama/Llama-3.1-8B"
# LORA_ADAPTER_PATH = "experiments/LLM_retrieval/llama3_qa_adapter_rq"
# INPUT_FILE = "data/qrecc/LLM_retrieval/test_passages_top100.jsonl"
# OUTPUT_FILE = "data/qrecc/LLM_retrieval/generated_queries.jsonl"

# For TopiOCQA:
# BASE_MODEL_PATH = "models/meta-llama/Llama-3.1-8B"
# LORA_ADAPTER_PATH = "experiments/LLM_retrieval/llama3_qa_adapter_topiocqa"
# INPUT_FILE = "data/topiocqa/LLM_retrieval/test_passages_top100.jsonl"
# OUTPUT_FILE = "data/topiocqa/LLM_retrieval/generated_queries.jsonl"

# For CAST:
# BASE_MODEL_PATH = "models/meta-llama/Llama-3.1-8B"
# LORA_ADAPTER_PATH = "experiments/LLM_retrieval/llama3_qa_adapter_topiocqa"
# INPUT_FILE = "data/cast/test_passages.jsonl"
# OUTPUT_FILE = "data/cast/generated_queries.jsonl"

# Default configuration (update these paths)
BASE_MODEL_PATH = "models/meta-llama/Llama-3.1-8B"
LORA_ADAPTER_PATH = "experiments/LLM_retrieval/llama3_qa_adapter"
INPUT_FILE = "data/test_passages.jsonl"
OUTPUT_FILE = "data/generated_queries.jsonl"


BATCH_SIZE = 64   # Adjust based on GPU memory
MAX_NEW_TOKENS = 50
NUM_QUERY = 5   # Number of queries to generate consecutively

# ==========================
# Load model and tokenizer
# ==========================
print("Loading base model (4bit quantization)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 is faster for inference
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"   # Decoder-only models must use left padding
# ==========================
# Prompt construction
# ==========================
def build_prompt(paragraph: str) -> str:
    system_prompt = "You are a meticulous AI assistant that functions as a question generator.Your goal is to carefully read the provided paragraph and create a single, clear, and concise question that can be fully answered using only the information in the paragraph."
    user_prompt =  f"Read the following paragraph carefully and generate one specific question that can be answered from the paragraph alone.\n\n**Paragraph:**\n{paragraph}"
    
    return f"system: {system_prompt}\nuser: {user_prompt}\nassistant:"

# ==========================
# Batch inference function
# ==========================
def generate_batch(paragraphs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_p=0.9):
    prompts = [build_prompt(p) for p in paragraphs]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    results = []
    for i in range(len(paragraphs)):
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[i][input_len:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        results.append(text)
    return results

# ==========================
# Process file
# ==========================
def process_paragraphs(input_file, output_file, batch_size=BATCH_SIZE):
    print(f"Starting file processing: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return

    # Read all paragraphs
    paragraphs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            paragraphs.append(data)

    print(f"Total paragraphs to process: {len(paragraphs)}")

    with open(output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(paragraphs), batch_size), desc="Generating queries"):
            batch = paragraphs[i:i+batch_size]
            p_ids = [d["p_id"] for d in batch]
            paras = [d["paragraph"] for d in batch]

            try:
                
                for i in range(NUM_QUERY):
                    queries = generate_batch(paras)
                    for p_id, query in zip(p_ids, queries):
                        fout.write(json.dumps({"p_id": p_id, "query": query}, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Batch {i} error: {e}")
                for p_id in p_ids:
                    fout.write(json.dumps({"p_id": p_id, "query": f"ERROR: {str(e)}"}, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Processing complete! Results saved to: {output_file}")

# ==========================
# Main function
# ==========================
if __name__ == "__main__":
    print("=" * 50)
    print("Fine-tuned model inference program (batch processing + quantization)")
    print("=" * 50)
    process_paragraphs(INPUT_FILE, OUTPUT_FILE, BATCH_SIZE)


# Example usage:
# CUDA_VISIBLE_DEVICES=0 nohup python src/retrieval/llama3_infer_a6000.py > logs/infer_output.out 2>&1 &