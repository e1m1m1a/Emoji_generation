# Emoji Generation
NLP project SNU fall semester 2022 

This project focuses on using NLP models to add emoji into text.

This repo contains the scripts to fine-tune 3 models for our task.

The pdf of the presentation used in our video is available in this repo under *presentation*.


## Dataset and preprocessing

The dataset used is *EmojifyData-EN: English tweets, with emojis* taken from Kaggle (https://www.kaggle.com/datasets/rexhaif/emojifydata-en).
It contains 18 million English tweets.

We used the data_preprocessing.py script of this repo to preprocess the data and generate the files used for training our models.

Therefore, to use our fine-tuning scripts, you should first download the dataset and execute the preprocessing script.


## Emoji prediction with RoBERTa


In this section we use cardiffnlp/twitter-roberta-base-dec2021 model from the Hugging Face (https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021) to create the classification model and we also the model "cardiffnlp/twitter-roberta-base-2021-124m-emoji" already trained on the emoji classification task (https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-emoji).
We finetune the first model on the emoji prediction task and we can use both models to predict emojis.

There is another script called emojional_eval.py that uses a special embedding to enable word2vec to support emojis and perform the emoji prediction task using a word2vec model.


### How to use

Once the dataset are created by the preprocessing script, you can use the *RoBERTa_classification/roberta_classification_model.py* script to fine-tune the model.

Then the evaluation script can be used to evaluate the model on the test set.


### Environment specifications

python: v 3.7.15

torch: v 1.13.0

transformers: v 4.24.0

tqdm: v 4.64.0

## Translation with T5


In this section we use different version of T5 model from the Hugging Face (https://huggingface.co/docs/transformers/model_doc/t5).
We finetune these models for the translation task between a language without emojis to a language with emojis.


### How to use

Once the dataset are created by the preprocessing script, you can use the *T5_translation/flan-t5-based_model* or *T5_translation/t5-v11_based_model* script to fine-tune the models.

Then the evaluation script can be used to evaluate the model on the test set.

### Environment specifications

python: v 3.7.15

torch: v 1.13.0

transformers: v 4.24.0

tqdm: v 4.64.0

## Mask prediction for emoji generation with BERT


In this section we use BERT-base-cased model from the Hugging Face (https://huggingface.co/bert-base-cased).
We add MASK token to the input sentences at every space or punctuation to let the model predict whether it's an empty string or an emoji, and which emoji.


### How to use


**To fine-tune the model:**

We will use the *BERT_mask_prediction/fine_tune_bert.py* script.

To fine-tune the model, you should first use the preprocessing script to generate the *input_data.txt* and *target_data.txt* files, and add them to the same folder that we'll call *dataset*.

Create an empty folder called *checkpoints*, and one called *best_model* at the location you will be running the script from. 

Then use the following command line *python fine_tune_bert.py --epochs=X --dataset_dir=path/to/dataset* (see script for --model_dir and --ckp_path optional arguments).

The checkpoints will be saved in the checkpoints folder.


**To make a prediction:**

We will use the *BERT_mask_prediction/make_prediction.py* script.

First create a file with your input tweets containing no emoji, and call it *input_data.txt*.

Use the following commad line *python make_prediction_bert.py --dataset_dir=path/to/the/directory/where/input_data.txt/is/stored* (see script for --model_dir and --decoding_type optional arguments).

If you already fine-tuned the model, you can use the checkpoint to make the prediction by adding the following optional argument to the commad line: *--ckp_path=path/to/checkpoint.pt*. Otherwise it will be run with BERT-base-cased pre-trained model.

The outputs will be saved in the current working directory under *predictions_outputs.txt*. 


### Environment specifications

python: v 3.7.15

torch: v 1.13.0

transformers: v 4.24.0

tqdm: v 4.64.0


#### Julian Paquerot - Emma Pruvost
