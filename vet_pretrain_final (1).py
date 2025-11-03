import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from transformers.trainer_utils import get_last_checkpoint

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model & Checkpoint Directory
model_name = "" 
checkpoint_dir = ""
resume_checkpoint = get_last_checkpoint(checkpoint_dir)

# Define max sequence length
max_seq_length = 2048

# Load Model and Tokenizer with Unsloth - much faster and memory-efficient
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=True  # Reduces memory usage by ~75% compared to fp16
)

# Apply Unsloth's optimized LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",  # "none" is optimized in Unsloth
    use_gradient_checkpointing="unsloth",  # Uses 30% less VRAM than standard gradient checkpointing
    random_state=42,
    max_seq_length=max_seq_length
)

# Load Dataset
print('Loading Dataset....')
#Put the dataset path from hf accordingly
full_dataset = load_dataset("")
print(len(full_dataset))
print('Dataset Loaded....')
dataset_size = len(full_dataset)



# Training Arguments optimized for Unsloth
training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    per_device_train_batch_size=8,  # Can increase batch size thanks to Unsloth's memory efficiency
    gradient_accumulation_steps=2,
    save_strategy="steps",
    save_steps=1000,
    logging_steps=5,
    num_train_epochs=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=3,
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False,  # Required for SFTTrainer
)

# Use SFTTrainer for simplified fine-tuning
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=full_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True  # Efficiently packs multiple sequences into a batch
)

# Resume training if checkpoint exists
trainer.train()

# Save the fine-tuned model
trainer.save_model()



