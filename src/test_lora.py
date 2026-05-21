import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


base_model_name = "distilgpt2"
adapter_path = "models/lora-support-agent"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

prompt = "Instruction: What is the PTO policy?\nResponse:"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
