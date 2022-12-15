# We are going to process the data for the emoji translator.
# We are going to use the data from the file: emojitweets-01-04-2018.txt
# We need two files, one with the tweets without emojis for the input of the model and one with the emojis for the target of the model.
# The target data will use tokens to represent the emojis.

# %% Imports
import emoji
import os
import pickle as pkl

# %% Globals
tweets_path = 'dataset/emojitweets-01-04-2018.txt'
clean_tweets_path = 'dataset/clean_tweets.txt'
input_data_path = 'dataset/input_data.txt'
target_data_path = 'dataset/target_data.txt'


# %% Input data
def create_input_data(raw_data_path, input_data_path, forced=False):
    """Create the input data file for the model by removing the emojis from the tweets.
    The superfluous spaces that may appear are also removed.

    Args:
        plain_data_path (str): Path to the plain data file.
        input_data_path (str): Path to the input data file. 
        forced (bool, optional): Force the creation of the file even if it already exists. Defaults to False.
    """
    # No need to process the file if it has already been done
    if forced or not os.path.isfile(input_data_path) or os.stat(input_data_path).st_size == 0:
        print('Creation of the input dataset, it may take several minutes')
        # Let's open the data file with utf-8 encoding
        with open(raw_data_path, 'r', encoding='utf-8') as raw_data:
            # Let's open the file for the tweets without emojis
            with open(input_data_path, 'w', encoding='utf-8') as input_data:
                # Let's read the original file line by line
                for line in raw_data:
                    line = emoji.replace_emoji(line, replace='')
                    # Let's delete double spaces
                    line.replace('  ', ' ')
                    # Let's delete spaces at the beginning of the line
                    if line[0] == ' ':
                        line = line[1:]
                    # And at the end of the line
                    if line[-1] == ' ':
                        line = line[:-1]
                    input_data.write(line)
                print('The input dataset has been created')
    else:
        print('The input dataset already exists')


# %% Target data
def create_target_data(raw_data_path, target_data_path, forced=False):
    """Create the target data file for the model by removing the emojis from the tweets.

    Args:
        plain_data_path (str): Path to the plain data file.
        target_data_path (str): Path to the target data file.
        forced (bool, optional): Force the creation of the file even if it already exists. Defaults to False.
    """

    # No need to process the file if it has already been done
    if forced or not os.path.isfile(target_data_path) or os.stat(target_data_path).st_size == 0:
        print('Creation of the target dataset, it may take several minutes')
        # Let's open the data file with utf-8 encoding
        with open(raw_data_path, 'r', encoding='utf-8') as raw_data:
            # Let's open the file for the tweets without tokens instead of emojis
            with open(target_data_path, 'w', encoding='utf-8') as target_data:
                # Let's read the original file line by line
                for line in raw_data:
                    target_data.write(emoji.replace_emoji(
                        line, replace=lambda chars, data_dict: data_dict['en']))
                print('The target dataset has been created')
    else:
        print('The target dataset already exists')


# %% Clean input data
def delete_spaces(data_path, clean_data_path):
    """Delete the superfluous spaces in the input data file.

        Superfluous spaces are the double spaces and those at the beginning or at the end of the line that are created when we remove the emojis from the data.

    Args:
        data_path (str): Path to the data file.
        clean_data_path (str): Path to the clean data file.
    """
    with open(data_path, 'r', encoding='utf-8') as data:
        with open(clean_data_path, 'w', encoding='utf-8') as data_clean:
            lines = data.readlines()
            for line in lines:
                # Let's delete double spaces
                line.replace('  ', ' ')
                # Let's delete spaces at the beginning of the line
                if line[0] == ' ':
                    line = line[1:]
                # And at the end of the line
                if line[-1] == ' ':
                    line = line[:-1]
                data_clean.write(line)
            print(f"The superfluous spaces of {data_path} have been deleted.")


def delete_dupes(data_path, clean_data_path):
    """Delete the duplicates in the data file.

    Args:
        data_path (str): Path to the data file.
        clean_data_path (str): Path to the clean data file.
    """
    with open(data_path, 'r+', encoding='utf-8') as data:
        with open(clean_data_path, 'w', encoding='utf-8') as data_no_dupes:
            lines = data.readlines()
            # Let's store the previously seen lines in a set to remove the duplicates more efficiently
            set_lines = set()
            for line in lines:
                if line not in set_lines:  # O(1) operation
                    set_lines.add(line)
                    data_no_dupes.write(line)
            print(f"The duplicates of {data_path} have been deleted.")


# %% Create token list
def save_tokens_list(data_file):
    """Create and save a list of all the tokens used in the data file.

    Args:
        data_file (str): Path to the data file.
    """
    print('Creation of the tokens lists, it may take several minutes')
    with open(data_file, 'r', encoding='utf-8') as data_file:
        emoji_list = emoji.distinct_emoji_list(data_file.read())
        tokens_list = [emoji.demojize(emoj) for emoj in emoji_list]
        print(f'The tokens lists have been created')

    with open('tokens_list.pkl', 'wb') as token_list_file:
        pkl.dump(tokens_list, token_list_file)


def save_tokens_list_light(data_file):
    """Create and save a list of all the tokens used in the data file.

    Args:
        data_file (str): Path to the data file.
    """
    print('Creation of the tokens lists (light), it may take several minutes')
    emoji_set = set()
    with open(data_file, 'r', encoding='utf-8') as data_file:
        for line in data_file:
            emoji_list = emoji.distinct_emoji_list(line)
            for emoj in emoji_list:
                emoji_set.add(emoji.demojize(emoj))
    tokens_list = list(emoji_set)
    with open('tokens_list.pkl', 'wb') as token_list_file:  # append
        pkl.dump(tokens_list, token_list_file)

# %% Main


def main():
    # Let's print the number of tweets in the dataset
    print(f'There are {len(open(tweets_path, "r", encoding="utf-8").readlines())} tweets in the raw dataset')

    # Let's delete delete dupes in the raw dataset
    delete_dupes(tweets_path, clean_tweets_path)
    print(f'There are {len(open(clean_tweets_path, "r", encoding="utf-8").readlines())} tweets in the clean dataset')

    # Let's create the input and target data
    create_input_data(clean_tweets_path, input_data_path)
    print(f'There are {len(open(input_data_path, "r", encoding="utf-8").readlines())} tweets in the input dataset')

    create_target_data(clean_tweets_path, target_data_path)
    print(f'There are {len(open(target_data_path, "r", encoding="utf-8").readlines())} tweets in the target dataset')

    # Let's save the tokens and emojis lists
    save_tokens_list(tweets_path)
    print(f"{len(pkl.load(open('tokens_list.pkl', 'rb')))} tokens in the token list")


if __name__ == '__main__':
    main()
