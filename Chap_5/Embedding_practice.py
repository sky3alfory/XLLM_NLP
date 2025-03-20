import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torch
import spacy

# from torchtext.legacy import data,datasets
from torchtext import *
from torchtext.datasets import IMDB
import torchtext 

print(torch.cuda.is_available())
print(spacy.require_gpu())

# nlp = spacy.load("en_core_web_sm")

# breakpoint()

# TEXT = legacy.data.Field(lower=True, include_lengths=True, batch_first=False, tokenize='spacy')
# LABEL = legacy.data.LabelField()

# print(type(datasets.IMDB()))

# train,test = datasets.IMDB()
# breakpoint()
# TEXT.build_vocab(train, vectors='glove.6B.100d')

# LABEL.build_vocab(train)

# train_iter, test_iter = data.BucketIterator.splits((train,test), batch_sizes=(128,1024), device = torch.device("cuda"), sort_within_batch=True, repeat=False)
print("GPU 사용 : ",torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

breakpoint()

train_list = list(train_iter)[:5]
for sample in train_list :
    print(sample)
print("hello")