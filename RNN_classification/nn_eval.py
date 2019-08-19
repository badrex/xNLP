# -*- coding: utf-8 -*-

from argparse import Namespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# own imports
import vectorization
import nn_models

args = Namespace(
    # Data and Path information
    frequency_cutoff=25,
    model_state_file='model.pth',
    document_csv='../data/news_documents/news_dataset.csv',
    # document_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='model_storage/cnn_doc_classification/',
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    vectorizer_file='vectorizer.json',
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
    use_glove=False,
    embedding_size=100,
    hidden_dim=100,
    num_channels=100,
    # Training hyper parameter
    seed=1337,
    learning_rate=0.001,
    dropout_p=0.1,
    batch_size=124,
)

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


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
    dataset = vectorization.NewsDataset.load_dataset_and_make_vectorizer(args.document_csv)
    dataset.save_vectorizer(args.vectorizer_file)

vectorizer = dataset.get_vectorizer()
# compute the loss & accuracy on the test set using the best available model

classifier = nn_models.NewsClassifier(embedding_size=args.embedding_size,
                            num_embeddings=len(vectorizer.title_vocab),
                            num_channels=args.num_channels,
                            hidden_dim=args.hidden_dim,
                            num_classes=len(vectorizer.category_vocab),
                            dropout_p=args.dropout_p,
                            padding_idx=0)

state_dict = torch.load(args.save_dir + args.model_state_file)
classifier.load_state_dict(state_dict)

classifier = classifier.to(args.device)
dataset.class_weights = dataset.class_weights.to(args.device)
loss_func = nn.CrossEntropyLoss(dataset.class_weights)

dataset.set_split('test')
batch_generator = vectorization.generate_batches(dataset,
                                   batch_size=args.batch_size,
                                   device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute the output
    y_pred =  classifier(batch_dict['x_data'])

    # compute the loss
    loss = loss_func(y_pred, batch_dict['y_target'])
    loss_t = loss.item()
    running_loss += (loss_t - running_loss) / (batch_index + 1)

    # compute the accuracy
    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_t - running_acc) / (batch_index + 1)



print("Test Accuracy: {:.2f}".format(running_acc))
