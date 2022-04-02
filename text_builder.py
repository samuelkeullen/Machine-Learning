import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation
import requests

#We are going to use a free downloadable book as the dataset for this tutorial: Alice’s Adventures in Wonderland by Lewis Carroll. But you can use any book/corpus you want.

content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
open("data/wonderland.txt", "w", encoding="utf-8").write(content)

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 30
# dataset file path
FILE_PATH = "data/wonderland.txt"
BASENAME = os.path.basename(FILE_PATH)
# read the data
text = open(FILE_PATH, encoding="utf-8").read()
# remove caps, comment this code if you want uppercase characters as well
text = text.lower()
# remove punctuation
text = text.translate(str.maketrans("", "", punctuation))

# print some stats
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(vocab)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(vocab)}

# save these dictionaries for later generation
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

# convert all text into integers
encoded_text = np.array([char2int[c] for c in text])

# construct tf.data.Dataset object
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

# print first 5 characters
for char in char_dataset.take(8):
    print(char.numpy(), int2char[char.numpy()])

# build sequences by batching
sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)

# print sequences
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))

# build sequences by batching
sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)

# print sequences
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))

def split_sample(sample):
    # example :
    # sequence_length is 10
    # sample is "python is a great pro" (21 length)
    # ds will equal to ('python is ', 'a') encoded as integers
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        # first (input_, target) will be ('ython is a', ' ')
        # second (input_, target) will be ('thon is a ', 'g')
        # third (input_, target) will be ('hon is a g', 'r')
        # and so on
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        # extend the dataset with these samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

split_sample("python is a great pro")

# prepare inputs and targets
dataset = sequences.flat_map(split_sample)

def one_hot_samples(input_, target):
    # onehot encode the inputs and the targets
    # Example:
    # if character 'd' is encoded as 3 and n_unique_chars = 5
    # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

# print first 2 samples
for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")