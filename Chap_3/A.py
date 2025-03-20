import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from thinc.api import require_gpu
import cupy
import spacy

data = pd.read_csv("/home/sky3alfory/NLP_prac/nlpbook/data/train.csv")
data = pd.DataFrame(data=data)

data.columns =data.columns.str.replace(" ","_")
data.columns = data.columns.str.lower()
data["class_name"] = data["class_index"].map({1:"World",2:"Sports",3:"Business",4:"Sci_Tech"})

cols = ["title","description"]
data[cols] = data[cols].applymap(lambda x: x.replace("\\"," "))
data[cols] = data[cols].applymap(lambda x: x.replace("#36","$"))
data[cols] = data[cols].applymap(lambda x: x.replace("  "," "))
data[cols] = data[cols].applymap(lambda x: x.strip())

data.to_csv("/home/sky3alfory/NLP_prac/nlpbook/data/train_prepared.csv", index=False)

if spacy.require_gpu() :
    print(" spaCy 패키지 GPU 사용가능")
else :
    print(spacy.require_gpu())
    
nlp = spacy.load("en_core_web_trf")

import pprint
pp = pprint.PrettyPrinter(indent=4)

for i in range(9):
    print("Article",i)
    print(data.loc[i,"description"])
    print("Text Start End Label")
    doc = nlp(data.loc[i,"description"])
    for token in doc.ents:
        print(token.text, token.start_char,token.end_char, token.label)
    print("\n")