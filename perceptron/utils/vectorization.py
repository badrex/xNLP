# -*- coding: utf-8 -*-

from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): a pre-existing map of tokens to indices
            add_unk (bool): a flag that indicates whether to add the UNK token
            unk_token (str): the UNK token to add into the Vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)


    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        return cls(**contents)

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """Add a list of tokens into the Vocabulary

        Args:
            tokens (list): a list of string tokens
        Returns:
            indices (list): a list of indices corresponding to the tokens
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)


class DocumentDataset(Dataset):
    def __init__(self, document_df, vectorizer):
        """
        Args:
            document_df (pandas.DataFrame): the dataset
            vectorizer (DocumentVectorizer): vectorizer instantiated from dataset
        """
        self.document_df = document_df
        self._vectorizer = vectorizer

        self.train_df = self.document_df[self.document_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.document_df[self.document_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.document_df[self.document_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, document_csv):
        """Load dataset and make a new vectorizer from scratch

        Args:
            document_csv (str): location of the dataset
        Returns:
            an instance of DocumentDataset
        """
        document_df = pd.read_csv(document_csv)
        train_document_df = document_df[document_df.split=='train']
        return cls(document_df, DocumentVectorizer.from_dataframe(train_document_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, document_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            document_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of DocumentDataset
        """
        document_df = pd.read_csv(document_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(document_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of DocumentVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return DocumentVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        document_vector = \
            self._vectorizer.vectorize(row.review)

        label_index = \
            self._vectorizer.label_vocab.lookup_token(row.rating)

        return {'x_data': document_vector,
                'y_target': label_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


# The Vectorizer

class DocumentVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, document_vocab, label_vocab):
        """
        Args:
            document_vocab (Vocabulary): maps words to integers
            label_vocab (Vocabulary): maps class labels to integers
        """
        self.document_vocab = document_vocab
        self.label_vocab = label_vocab

    def vectorize(self, review):
        """Create a collapsed one-hit vector for the review

        Args:
            review (str): the review
        Returns:
            one_hot (np.ndarray): the collapsed one-hot encoding
        """
        one_hot = np.zeros(len(self.document_vocab), dtype=np.float32)

        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.document_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, document_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            document_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the DocumentVectorizer
        """
        document_vocab = Vocabulary(add_unk=True)
        label_vocab = Vocabulary(add_unk=False)

        # Add ratings
        for rating in sorted(set(document_df.rating)):
            label_vocab.add_token(rating)

        # Add top words if count > provided count
        word_counts = Counter()
        for review in document_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                document_vocab.add_token(word)

        return cls(document_vocab, label_vocab)

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a DocumentVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the DocumentVectorizer class
        """
        document_vocab = Vocabulary.from_serializable(contents['document_vocab'])
        label_vocab =  Vocabulary.from_serializable(contents['label_vocab'])

        return cls(document_vocab=document_vocab, label_vocab=label_vocab)

    def to_serializable(self):
        """Create the serializable dictionary for caching

        Returns:
            contents (dict): the serializable dictionary
        """
        return {'document_vocab': self.document_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}
