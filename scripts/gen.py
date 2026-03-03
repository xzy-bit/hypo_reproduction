import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

#model_path = "results/model/llama3-8b_dpo/"          # 替换为您的模型路径
model_path = "results/model/lr2_beta2/"
output_file = "responses/lr2_beta2.json"

print("Loading AlpacaEval dataset...")
eval_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Initializing vLLM...")
llm = LLM(
    model=model_path,
    #model="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer="meta-llama/Meta-Llama-3-8B-Instruct",         # 使用模型自带的 tokenizer 配置
    trust_remote_code=True,
    tensor_parallel_size=4,        # 根据 GPU 数量调整
    dtype="bfloat16",               # 与训练时保持一致
)

sampling_params = SamplingParams(
    temperature=0.9,                # 论文要求 Llama-3-8B 为 0.9
    top_p=1.0,                     # 可选，通常与采样配合使用，默认 1.0 也可
    max_tokens=4096,                 # 最大生成长度，可根据需要调整
    stop_token_ids=[128001,128009],  # 遇到 eos 停止
)


prompts = [item["instruction"] for item in eval_dataset]

print(f"Generating {len(prompts)} responses...")
outputs = llm.generate(prompts, sampling_params)

results = []
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text.strip()
    results.append({
        "instruction": eval_dataset[i]["instruction"],
        "output": generated_text,
        "generator": "llama3.1-8b_hypo-dpo"  # 替换为您的模型名称（将显示在 leaderboard 上）
    })

# 保存为 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done! Results saved to {output_file}")
