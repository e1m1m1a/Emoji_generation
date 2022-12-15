# %% Imports
import os
import pickle as pkl
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
from datasets.load import load_dataset, load_from_disk
from datasets.combine import concatenate_datasets
from transformers import AutoTokenizer
from functools import partial


# %% Globals
forced = False  # Force the creation of the tokenizer and datasets even if they already exist
model_version = '0.1'

test_size = 0.2
model_max_length = 200

model_names = {'small': 'flan-t5-small',
               'base': 'flan-t5-base',
               'large': 'flan-t5-large',
               'xl': 'flan-t5-xl',
               'xxl': 'flan-t5-xxl'}
model_size = 'base'
model_name = model_names[model_size]
model_path = f"google/{model_name}"

input_data_path = 'dataset/input_data.txt'
target_data_path = 'dataset/target_data.txt'

hf_dataset_path = 'hf_dataset'
tokenized_dataset_path = model_name + '_hf_tokenized_dataset'
tokenizer_path = model_name + '_emoji_tokenizer'
output_path = f"{model_name}_v_{model_version}_results"
evaluation_strategy = 'epoch'

resume_from_checkpoint = None
save_path = output_path + '/final_model'

# Training hyperparameters
learning_rate = 2e-5
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
weight_decay = 0.01
save_total_limit = 3
num_train_epochs = 1

prefix = 'Inject emoji in this text : '  # Prefix for the input text
emoji_tokens = pkl.load(open('tokens_list.pkl', 'rb')) + ['emoji']  # Load the new tokens and add emoji because it may not be in the vocabulary and I think it's important

# emoji_tokens = [emoji for emoji in emoji_tokens if emoji != 'ℹ'] # Delete this character: ℹ


# %% Model and Tokenizer
def add_tokens_to_tokenizer(token_list, tokenizer):
    """Add new tokens to the tokenizer.

    Args:
        token_list (list): The list of new tokens to add.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer to add the tokens to.
    """
    # Let's check if the tokens are already in the vocabulary (spoiler: some are)
    new_tokens = set(token_list) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_tokens))


# %% Untokenized dataset
def process_dataset(input_data_path, target_data_path):
    """Process the dataset and save it as a huggingface dataset dictionnary.

    Args:
        input_data_path (str): The path to the input text file.
        target_data_path (str): The path to the target text file.

    Returns:
        datasets.dataset_dict.DatasetDict: The huggingface dataset dictionnary.
    """
    print('Processing the dataset...')
    # First load the input dataset and rename the column
    input_dataset = load_dataset('text', data_files={'train': input_data_path})
    input_dataset = input_dataset.rename_column('text', 'input_text')

    # Same with the target dataset
    target_dataset = load_dataset('text', data_files={'train': target_data_path})
    target_dataset = target_dataset.rename_column('text', 'target_text')

    # Concatenate the two datasets
    train_set = concatenate_datasets([input_dataset['train'], target_dataset['train']], axis=1)

    return train_set.train_test_split(test_size=test_size)


# %% Tokenize the dataset
def preprocess_function(examples, tokenizer):
    """Tokenize the input and target text with the prefix added at the beggining of the input text.

    Args:
        examples (dict): The dataset dictionary.

    Returns:
        dict: The tokenized dataset dictionary.
    """
    inputs = [prefix + ex for ex in examples["input_text"]]

    return tokenizer(text=inputs, text_target=examples['target_text'], truncation=True, max_length=model_max_length)


# %% Main
def main():
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Load the tokenizer
    # Check if the tokenizer already exists
    if not os.path.exists(tokenizer_path) or forced:
        tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length)

        # Add the new tokens to the tokenizer and save
        add_tokens_to_tokenizer(emoji_tokens, tokenizer)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)

    # Let's add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    print('Tokenizer loaded')

    # Load the dataset (not tokenized yet)
    # Check if the dataset already exists
    if not os.path.exists(hf_dataset_path) or forced:
        # Process the dataset and save
        dataset = process_dataset(input_data_path, target_data_path)
        dataset.save_to_disk(hf_dataset_path)
    else:
        dataset = load_from_disk(hf_dataset_path)

    print('Untokenized dataset loaded')

    # Load the tokenized dataset
    # Check if the tokenized dataset already exists
    if not os.path.exists(tokenized_dataset_path) or forced:
        # Tokenize the dataset and save
        tokenized_dataset = dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True)  # The partial function is used to pass the tokenizer to the preprocess_function
        tokenized_dataset.save_to_disk(tokenized_dataset_path)
    else:
        tokenized_dataset = load_from_disk(tokenized_dataset_path)

    print('Tokenized dataset loaded')

    # Load the data collector, trainer and training arguments
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        evaluation_strategy=evaluation_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        fp16=True,  # Uncomment this if you have a GPU with CUDA
        # remove_unused_columns=False
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Finally train the model and save
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(save_path)


if __name__ == '__main__':
    main()

# %%
