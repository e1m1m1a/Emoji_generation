from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch
import shutil
from tqdm import tqdm
import argparse
import os


# =============================================================================
# TO DO BEFORE RUNNING
# checkpoints and best_model directories should be created in the folder from where the script is run
# =============================================================================


# =============================================================================
# INPUT PARAMETERS
# =============================================================================

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument('--epochs', required=True, type=int,
                    help='Number of epochs to train the model for.')
parser.add_argument('--dataset_dir', required=True, type=str,
                    help='Directory containing input_data.txt and target_data.txt files from the pre-processing.')

# Optional parameters
parser.add_argument('--model_dir', required=False, type=str, default="",
                    help='Directory containing the model if not loaded directly from the hugging face.')
parser.add_argument('--ckp_path', required=False, type=str, default="",
                    help='Checkpoint path if should start the training from a checkpoint (.pt file).')

args = parser.parse_args()
globals().update(args.__dict__)


# =============================================================================
# UTILITIES
# =============================================================================

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def save_ckp(state, is_best, checkpoint_path, best_model_dir):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + '/best_model.pt'
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


# =============================================================================
# LOAD MODEL, TOKENIZER, OPTIMIZER
# =============================================================================

if model_dir != "":
    model = BertForMaskedLM.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
else:
    model = BertForMaskedLM.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # using bert-base-cased by default (bert-large-cased was too heavy)

# deprecated (use -> torch.optim.AdamW in the future)
optimizer = AdamW(model.parameters(), lr=10e-4, no_deprecation_warning=True)

start_epoch = 0  # assum fine-tuning on model is run for the first time

if ckp_path != "":
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
    print("Start training from checkpoint at epoch" + str(start_epoch))

# add special token [EMPTY] to be used as empty string predicitons
tokenizer.add_tokens(['[EMPTY]'], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))


# =============================================================================
# PROCESS DATA
# =============================================================================

# load tweets and store into arrays

with open(dataset_dir+'/input_data.txt', 'r') as f_input:
    input_tweets = f_input.read().replace(" ", "[MASK] ").replace("\n", "[MASK]\n").split('\n')
print("training on 150000 lines of input file out of " +
      str(len(input_tweets)) + " lines")
# select only few lines in the dataset due to computation limitation
input_tweets = input_tweets[:150000]
print("input tweets loaded")

with open(dataset_dir+'/target_data.txt', 'r') as f_target:
    # select only few lines in the dataset due to computation limitation
    target_tweets = f_target.read().replace(
        " ", "[EMPTY] ").split('\n')[:150000]
print("label tweets loaded")

# tokenize tweets arrays (create dict with input_ids, attention_mask and labels keys containing torch tensors)

inputs = tokenizer(input_tweets, return_tensors='pt',
                   max_length=128, truncation=True, padding='max_length')
print("inputs tokenized")
labels = tokenizer(target_tweets, return_tensors='pt',
                   max_length=128, truncation=True, padding='max_length')
print("labels tokenized")

# change labels tokens ids from the inputs to their actual labels token ids

inputs['labels'] = labels.input_ids.detach().clone()

# set all ids from token that won't be predicted to -100 so that these tokens won't be used in loss calculation

empty_token_idx = tokenizer.convert_tokens_to_ids(["[EMPTY]"])[0]
for l in range(len(inputs['labels'][0])):
    if (inputs['labels'][0][l] != empty_token_idx) and (inputs['labels'][0][l] != 103) and (inputs['labels'][0][l] != 0) and (inputs['labels'][0][l] != 101) and (inputs['labels'][0][l] != 102):
        inputs['labels'][0][l] = -100
print("labels processed and associated to corresponding inputs")

# create dataset from inputs dict

dataset = MeditationsDataset(inputs)
print("dataset created")

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
print("loader initilized")


# =============================================================================
# TRAINING
# =============================================================================

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

for epoch in range(start_epoch, start_epoch+epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optimizer.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optimizer.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
    
        # save model after each epoch and keep best model according to loss on final batch from each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss
        }
        is_best = True
        if os.path.isfile('./best_model/best_model.pt'):
            current_best = torch.load("./best_model/best_model.pt")
            if current_best['loss'] < loss:
                is_best = False
        save_ckp(checkpoint, is_best, "./checkpoints/checkpoint_" +
                 str(epoch) + ".pt", "./best_model")
