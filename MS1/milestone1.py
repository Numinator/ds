from secrets import token_bytes
import numpy as np
import pandas as pd
import nltk 
import re
import nltk

def clean_text(txt):
  txt = txt.lower()
  txt = re.sub(r"\s+", " ", txt)
  regex = {"DATE" : r"(\d+[\\/-])?\d+[\\/-]\d+",
      "EMAIL" : r"\w+@\w+\.[a-z]{2,}",
      "URL" : r"([a-z\d]+://)?[a-z]+[a-z\d]*(\.[a-z\d\-\._~]+)+(/[a-z\d\-\._~]*|(\?[a-z\d\-\._~]+=[a-z\d\-\._~]+))*",
      "NUM" : r"[+-]?((\d+(,|\.))*\d+)"
      
  }
  for k in regex.keys():
    txt = re.sub(regex[k], k, txt) 
  return txt


text = "  Bla sammen-sat 2019-22-01 Bla. bla, ?  +231 123 123.4 123,4 122.112,2 122,112.2  bla  gg@no.re   bla\nnew\t dr.dk dr.Einstein www.youtube.com youtube.com youtube.dk https://www.youtube.com/ \ttab  "


df = pd.read_csv("https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv", delimiter=",")
df_content = df["content"]
df_cleaned = df_content.apply(clean_text)

def tokenize(txt):
  tokens = nltk.word_tokenize(txt)
  prepostfix = lambda t: "<"+t+">" if t.isupper() else t
  return list(map(prepostfix, tokens))

df_token = df_cleaned.apply(tokenize)

print(df_content.iloc[7])
print(df_cleaned.iloc[7])
print(df_token.iloc[7])




