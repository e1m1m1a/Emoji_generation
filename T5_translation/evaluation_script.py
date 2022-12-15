# %% Imports
# Huggingface
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets.load import load_from_disk

# Others
import torch
import numpy as np
import emoji


# %% Globals
model_name = 'base'
model_version = '0.1'
model_path = f"{model_name}_results/v_{model_version}"
dataset_path = f"{model_name}_hf_tokenized_dataset"

# model_path = "t5-v1_1-base_v_from_scratch.0.1_results/final_model"
# dataset_path = "t5-v1_1-base_hf_tokenized_dataset"

model_path = "flan-t5-base_v_0.1_results/checkpoint-46500"
dataset_path = "flan-t5-base_hf_tokenized_dataset"


model_max_length = 140
prefix = 'Inject emoji in this text : '  # Prefix for the input text
# prefix = ''

sample_size = 20


# %% Load model and stuffs
# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length)


def model_pred(text, model=model, tokenizer=tokenizer, emojize=True):
    encoding = tokenizer(text, return_tensors='pt')
    output = model.generate(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"], max_length=model_max_length)
    output = tokenizer.decode(output[0])
    return output


# Load dataset
dataset = load_from_disk(dataset_path)['test']
input_data = dataset['input_text']
tokenized_input_data = dataset['input_ids']
target_data = dataset['target_text']


# %% Let's perform the inference
predictions = []
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id

for i in range(sample_size):
    tens_data = torch.tensor(tokenized_input_data[i]).unsqueeze(0)
    pred = model.generate(tens_data, max_length=model_max_length)[0]
    # Delete the pad and eos tokens
    if pred[-1] == eos_token_id:
        pred = pred[:-1]
    if pred[0] == pad_token_id:
        pred = pred[1:]
    pred = tokenizer.decode(pred)
    predictions.append(pred)
    if i % 10 == 0:
        print(f"Progress: {i}/{sample_size}")

print("Predictions done")


# %% Compute BLEU score
bleu = load("bleu")

scores = bleu.compute(predictions=predictions, references=target_data[:sample_size])

print(scores['bleu'])

# %% Compute reference BLEU score
bleu = load("bleu")

scores = bleu.compute(predictions=input_data[:sample_size], references=target_data[:sample_size])

print(scores['bleu'])
# %% Comparison
full_comparison = [[predictions[i], target_data[i]] for i in range(sample_size)]

# %% Print all results
for i in range(sample_size):
    print(f"Input: {input_data[i]}")
    print(f"Target: {target_data[i]}")
    print(f"Prediction: {predictions[i]}")
    print('-------------------------------------------')
# %%
# %%
