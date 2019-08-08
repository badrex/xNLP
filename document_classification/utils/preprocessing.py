# -*- coding: utf-8 -*-

import collections
import numpy as np
import pandas as pd
import re
import csv
import random
import argparse

# toktok tokenizer
from nltk.tokenize import ToktokTokenizer


def preprocess_document(text):
    """Given text, preprocess, tokenize, then detokenize. """
    # instantiate toktok tokenizer
    toktok = ToktokTokenizer()

    return tokenize_then_detokenize(toktok, text)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text

def tokenize_then_detokenize(tokenizer, text):
    return normalize_text(' '.join(tokenizer.tokenize(text.lower().strip())))



def make_dataset():
    """
    Read file names from user input, and generate dataset.
    """

    DISCR = 'Generate dataset from input files to one csv frame.'

    parser = argparse.ArgumentParser(description=DISCR)

    # USER ARGS
    parser.add_argument('-train',
        type=str,
        help='Path to training dataset.',
        required=True)

    parser.add_argument('-test',
        type=str,
        help='Path to test dataset.',
        required=True)

    parser.add_argument('-output',
        type=str,
        help='Path of the output csv file.',
        required=True)


    user_args = parser.parse_args()

    # ARGs
    args = argparse.Namespace(
        raw_train_dataset_csv=user_args.train,
        raw_test_dataset_csv=user_args.test,
        proportion_subset_of_train=0.1,
        train_proportion=0.9,
        val_proportion=0.1,
        test_proportion=0.0,
        output_munged_csv=user_args.output,
        seed=1337
    )

    # Read training data
    train_documents = pd.read_csv(args.raw_train_dataset_csv,
        header=None,
        sep='","',
        engine='python',
        names=['label', 'text'])

    # Read test data
    test_documents = pd.read_csv(args.raw_test_dataset_csv,
        header=None,
        sep='","',
        engine='python',
        names=['label', 'text'])


    np.random.shuffle(train_documents.values)


    print(train_documents.head())
    print('train:', train_documents.label.value_counts())
    print('test:', test_documents.label.value_counts())
    #print(set(train_documents.label))

    # Create split data as a single data structure
    final_dataset = list()

    for i, row in train_documents.iterrows():
        entry = row.to_dict()

        if i < args.train_proportion*len(train_documents):
            entry['split'] = 'train'

        else:
            entry['split'] = 'val'

        final_dataset.append(entry)

    for _, row in test_documents.iterrows():
        entry = row.to_dict()
        entry['split'] = 'test'
        final_dataset.append(entry)

    final_dataset = pd.DataFrame(final_dataset)

    print(final_dataset.split.value_counts())


    final_dataset.text = final_dataset.text.apply(preprocess_document)


    lable_dict = {
        '"1"': 'World',
        '"2"': 'Sports',
        '"3"': 'Business',
        '"4"': 'Sci/Tech',
    }

    final_dataset['label'] = final_dataset.label.apply(lable_dict.get)

    print(final_dataset.head())

    final_dataset.to_csv(args.output_munged_csv, index=False)

def main():
    make_dataset()

if __name__ == '__main__':
    main()
