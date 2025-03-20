import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torch 
import spacy

from torchtext import *
import torchtext 

# nlp = spacy.load("en_core_web_sm")

# breakpoint()

TEXT = legacy.data.Field(lower=True, include_lengths=True, batch_first=False, tokenize='spacy')
LABEL = data.LabelField

train,test = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train, vectors='glove.6B.100d')

LABEL.build_vocab(train)

train_iter, test_iter = data.BucketIterator.splits((train,test), batch_sizes=(128,1024), device = torch.device("cuda"), sort_within_batch=True, repeat=False)