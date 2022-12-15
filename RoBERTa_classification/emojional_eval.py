# %% Imports

from gensim.models import Word2Vec
import numpy as np
import emoji
import os
import gensim
import gensim.downloader as api
import phrase2vec as p2v
import pickle as pkl
from evaluate import load
from random import randint

# Huggingface
from datasets.load import load_from_disk

# %% Globals
dataset_path = 'hf_dataset'
sample_size = 100

# %% Loads
# Load emoji2vec model and word2vec model
e2v = gensim.models.KeyedVectors.load_word2vec_format("emojional.bin", binary=True)
w2v = api.load('word2vec-google-news-300')
model = p2v.Phrase2Vec(300, w2v=w2v, e2v=e2v)

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
        output += f"{sentence} {best_emoji_sentence(sentence, model=model)} "
    final_emoj = best_emoji_sentence(text, model=model)
    rnd = randint(0, 2)
    for i in range(rnd):
        output += f"{final_emoj} "
    return output


def best_emoji_sentence(sentence, model=model):
    """Return the emoji that best fits the sentence."""
    vect = model[sentence]
    emoj = model.emojiVecModel.most_similar([vect], topn=1)
    return emoj[0][0]


def best_single_emoji(sentence, model=model):
    """Return the emoji that best fits the sentence."""
    best_emoj = None
    best_score = None
    for word in sentence.split():
        vect = model[word]
        emoj = model.emojiVecModel.most_similar([vect], topn=1)
        if best_score is None or emoj[0][1] > best_score:
            best_score = emoj[0][1]
            best_emoj = emoj[0][0]
    return best_emoj


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
