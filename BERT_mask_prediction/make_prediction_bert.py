from transformers import BertTokenizer, BertForMaskedLM
import torch
import random
import argparse
import os


# =============================================================================
# INPUT PARAMETERS
# =============================================================================

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument('--dataset_dir', required=True, type=str,
                    help='Directory containing input_data.txt (and eventually the target_data.txt), ie the files of the pre-processed tweets to generate emojis for.')

# Optional parameters
parser.add_argument('--ckp_path', required=False, type=str, default="",
                    help='Path to the checkpoint of the best model (.pt file).')
parser.add_argument('--model_dir', required=False, type=str, default="",
                    help='Directory containing the model if not loaded directly from the hugging face.')
parser.add_argument('--decoding_type', required=False, type=str, default="random",
                    help='Whether to predict the emojis from left to right or randomly in the tweet (can be random or left to right).')

args = parser.parse_args()
globals().update(args.__dict__)


# =============================================================================
# UTILITIES
# =============================================================================

def load_ckp(checkpoint_fpath, model):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


# =============================================================================
# LOAD MODEL, TOKENIZER
# =============================================================================

if model_dir != "":
    model = BertForMaskedLM.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
else:
    model = BertForMaskedLM.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # if ckp_path arg indicated, default model should be the same as the one used to generate the indicated checkpoint

tokenizer.add_tokens(['[EMPTY]'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

if ckp_path != "":
    model = load_ckp(ckp_path, model)


# =============================================================================
# PROCESS DATA
# =============================================================================

with open(dataset_dir+'/input_data.txt', 'r') as f_input:
    input_tweets = f_input.read().replace(" ", "[MASK] ").replace("\n", "[MASK]\n").split('\n')
    input_len = len(input_tweets)
    input_tweets = input_tweets[input_len-100:]  # select end of the dataset (wasn't used in the training)
    
inputs = tokenizer(input_tweets, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length')
    
# if available for testing, load labels of the input tweets to predict

if os.path.isfile(dataset_dir+'/target_data.txt'): 
    
    with open(dataset_dir+'/target_data.txt', 'r') as f_target:
        target_tweets = f_target.read().replace(" ", "[EMPTY] ").split('\n')
        target_len = len(target_tweets)
        target_tweets = target_tweets[target_len-100:]  # select end of the dataset (wasn't used in the training)

    labels = tokenizer(target_tweets, return_tensors='pt',
                       max_length=512, truncation=True, padding='max_length')
    
    inputs['labels'] = labels.input_ids.detach().clone()

print("data loaded")

# =============================================================================
# PREDICTION
# =============================================================================

model.eval()

# create file with all predictions
with open('prediction_outputs.txt', 'w') as f:
    f.write('')
print("prediction file created")

# loop through all tweets
for input_ids in inputs.input_ids:
    
    # while there's still a MASK token to predict in the input tweet
    while 103 in input_ids:

        # get indexes of the MASK ids
        mask_idxs = duplicates(input_ids, 103)

        # define next MASK to predict
        if decoding_type == 'left to right':
            focus_mask_idx = min(mask_idxs)
        else:
            focus_mask_idx = random.choice(mask_idxs)

        # remove the input index of the MASK to be predicted from mask_idxs
        mask_idxs.pop(mask_idxs.index(focus_mask_idx))

        # create array with the token ids of all input_ids elements excluding the MASK not to be predicted
        temp_indexed_tokens = [ids for idx,
                               ids in enumerate(input_ids) if idx not in mask_idxs]
        
        # create array of one element being the index of the mask to be predicted inside temp_indexed_tokens array
        ff = [i for i, ids in enumerate(temp_indexed_tokens) if ids == 103]
        
        # create tensor with 0 values everywhere
        temp_segments_ids = [0]*len(input_ids)
        segments_tensors = torch.tensor([temp_segments_ids])

        # make prediction
        with torch.no_grad():
            outputs = model(torch.unsqueeze(input_ids, 0),
                            token_type_ids=segments_tensors)
            predictions = outputs[0]

        # select one candidate among the predictions by doing top k-sampling
        k = 5
        predicted_index = random.choice(
            predictions[0, ff].argsort()[0][-k:]).item()

        # replace MASK by predicted ids
        input_ids[focus_mask_idx] = predicted_index

    # once all MASK in a tweet have been predicted, convert all tweet ids to tokens and then to words
    token_prediction = tokenizer.convert_ids_to_tokens(
        input_ids, skip_special_tokens=False)
    tweet_prediction = tokenizer.convert_tokens_to_string(token_prediction).replace(" .", ".").replace(" !", "!").replace(
        " ?", "?").replace(" ,", ",").replace("  ", " ").replace(" [PAD]", "").replace("[SEP]", "").replace("[CLS]", "").replace("[EMPTY]", "").replace("  ", " ")
    
    # store input tweet and corresponding prediction in file
    with open("prediction_outputs.txt", "a") as f:
        f.write("\n" + tweet_prediction)
    #print(tweet_prediction)
print("all predictions done")
