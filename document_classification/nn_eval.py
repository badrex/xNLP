# -*- coding: utf-8 -*-

from argparse import Namespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# own imports
from utils import vectorization
from nn_models import MLPerceptron

args = Namespace(
    # Data and Path information
    frequency_cutoff=25,
    model_state_file='model.pth',
    document_csv='../data/news_documents/news_dataset.csv',
    # document_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='model_storage/perceptron_classifier/',
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    vectorizer_file='vectorizer.json',
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    batch_size = 124
)


# Check CUDA
if not torch.cuda.is_available(): args.cuda = False

print("Using CUDA: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")


if args.reload_from_files:
    # training from a checkpoint
    print("Loading dataset and vectorizer")
    dataset = vectorization.DocumentDataset.load_dataset_and_load_vectorizer(args.document_csv,
                                                            args.vectorizer_file)
else:
    print("Loading dataset and creating vectorizer")
    # create dataset and vectorizer
    dataset = vectorization.DocumentDataset.load_dataset_and_make_vectorizer(args.document_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()


# compute the loss & accuracy on the test set using the best available model
classifier = MLPerceptron(input_dim=len(vectorizer.document_vocab), hidden_dim=100, output_dim=4)

classifier.load_state_dict(torch.load(args.save_dir + args.model_state_file))
classifier = classifier.to(args.device)

dataset.set_split('test')
batch_generator = vectorization.generate_batches(dataset,
                                   batch_size=args.batch_size,
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred = classifier(x_in=batch_dict['x_data'])

    # compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)

print("Test Accuracy: {:.2f}".format(running_acc))


# Sort weights
fc1_weights = classifier.fc1.weight.detach()[0]
_, indices = torch.sort(fc1_weights, dim=0, descending=True)
indices = indices.cpu().numpy().tolist()

# Top 20 words
print("Influential words in Positive Reviews:")
print("--------------------------------------")
for i in range(20):
    print(vectorizer.document_vocab.lookup_index(indices[i]))

print("====\n\n\n")

# Top 20 negative words
print("Influential words in Negative Reviews:")
print("--------------------------------------")
indices.reverse()
for i in range(20):
    print(vectorizer.document_vocab.lookup_index(indices[i]))
