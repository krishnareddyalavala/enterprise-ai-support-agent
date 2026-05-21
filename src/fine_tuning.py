from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model


model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


dataset = load_dataset(
    "json",
    data_files="data/support_finetune.jsonl"
)


def tokenize_function(example):

    text = f"Instruction: {example['instruction']}\nResponse: {example['response']}"

    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=["instruction", "response"]
)

training_args = TrainingArguments(
    output_dir="models/lora-support-agent",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=10,
    learning_rate=2e-4,
    fp16=False,
    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)


print("Starting LoRA fine-tuning...")

trainer.train()

print("Training completed.")


model.save_pretrained("models/lora-support-agent")

tokenizer.save_pretrained("models/lora-support-agent")

print("LoRA model saved to models/lora-support-agent")