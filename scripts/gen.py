import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

model_path = "results/model/llama3-8b_sp_dpo/"
output_file = "responses/sp_dpo.json"

print("Loading AlpacaEval dataset...")
eval_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction)

prompts = [format_prompt(item["instruction"]) for item in eval_dataset]

print("Initializing vLLM...")
llm = LLM(
    model=model_path,
    tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=4,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0.9,
    top_p=1.0,
    max_tokens=4096,                
    stop_token_ids=[128001, 128009]
)

print(f"Generating {len(prompts)} responses...")
outputs = llm.generate(prompts, sampling_params)

results = []
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text.strip()
    results.append({
        "instruction": eval_dataset[i]["instruction"],
        "output": generated_text,
        "generator": "llama3.1-8b-sp_dpo"
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done! Results saved to {output_file}")
