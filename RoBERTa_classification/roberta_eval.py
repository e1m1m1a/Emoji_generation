#%% Imports
import tweetnlp

# Huggingface
from datasets.load import load_from_disk

from evaluate import load
from random import randint


# %% Globals
dataset_path = 'hf_dataset'
sample_size = 100


#%% Load models
model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-2021-124m-emoji", max_length=128)

# Load dataset
dataset = load_from_disk(dataset_path)['test']
input_data = dataset['input_text']
target_data = dataset['target_text']


# %% Useful functions
def split_sentences(text):
    """Split a text into sentences.
    The sentences are split at the following characters: '.', '?', '!', '\n'."""
    split_characters = ['.', '?', '!', '\n']
    sentences = []
    sentence = ''
    for i, char in enumerate(text):
        if i == len(text) - 1:
            sentences.append(f"{sentence}{char}")
        elif char in split_characters and text[i + 1] not in split_characters:
            sentences.append(f"{sentence}{char}")
            sentence = ''
        else:
            sentence += char
    return sentences


# %%
def model_pred(text, model=model):
    """Perform the inference on a text."""
    sentences = split_sentences(text)
    output = ''
    for sentence in sentences:
        output += f"{sentence} {model.predict(sentence)['label']} "
    final_emoj = model.predict(text)['label']
    rnd = randint(0, 2)
    for i in range(rnd):
        output += f"{final_emoj} "
    return output


# %% Let's perform the inference
predictions = []

for i in range(sample_size):
    sample = input_data[i]
    pred = model_pred(sample)
    predictions.append(pred)
    if i % 10 == 0:
        print(f"Progress: {i}/{sample_size}")

print("Predictions done")


# %% Compute BLEU score
bleu = load("bleu")

scores = bleu.compute(predictions=predictions, references=target_data[:sample_size])

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
