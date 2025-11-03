#Initialize the libraries
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from torch import nn
from transformers.models.llama.modeling_llama import *
from trl import SFTTrainer
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments,AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login
import wandb

#Please use your own
WANDB_TOKEN = ""
HF_TOKEN = ""

wandb.login(key=WANDB_TOKEN)
login(token=HF_TOKEN)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
#Pu the output path
output_path = ""


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

#Model and Tokenizer loading
model, tokenizer = FastLanguageModel.from_pretrained(
    #Need to put the pretain model path
    model_name = " ",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf 
# If you're using a model that requires authentication
)

# # TOKENS_CONFIG = "havocy28/VetBERT"
# HEADS = 6 # @param {type: "number"}
# DIMENSIONS = 768 # @param {type: "number"}
# LAYERS = 6 # @param {type: "number"}
# INTERMEDIATE_SIZE= 1024 # @param {type: "number"}
CONTEXT_LENGTH = 256 # @param {type:"number"}
# NEW_MODEL = "Finetuned_llama_3.2_1B" # @param {type:"string"}


#Please put the SFT dataset path
DATASET = " " # @param {type:"string"}
# BATCH_SIZE = 32 # @param {type:"number"}
# LEARNING_RATE = 1e-5 # @param {type:"number"}
# EPOCHS = 10 # @param {type:"number"}

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


data = load_dataset(DATASET, split = "train",token=HF_TOKEN)



#data = load_dataset(DATASET, split = "train",token=HF_TOKEN)

#tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# Helper function for prompt formatting
def tokenize(data_set):
    instructions = data_set["instruction"]
    inputs= data_set["input"]
    outputs= data_set["output"]
    texts = []
    for ins, inp, outp in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(ins, inp, outp) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


# Apply tokenization to the dataset
tokenized_data = data.map(tokenize, batched=True)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset =tokenized_data,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 60,
        num_train_epochs=5,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_path,
        report_to = "wandb", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()