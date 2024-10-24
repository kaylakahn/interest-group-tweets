import os
import pandas as pd
from transformers import pipeline
import torch # if on apple, must be imported or model will fall to cpu

# load df
# remove NA rows
# these are not missing data, just extra rows from each csv that was compiled
df = pd.read_csv("/russell_tweets_oct2424.csv")
df = df[df['text'].notna()]

# define hypothesis and labels
candidate_labels = ["takes a position", "does not take a position"]
hypothesis_template = "This tweet {} on a political issue"
# deberta-v3 zeroshot pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=torch.device('mps'), batch_size = 32)

# classify 'text' column
results = classifier(list(df['text']), candidate_labels=candidate_labels, hypothesis_template=hypothesis_template, multi_label=False)
df['predicted_label'] = [result['labels'][0] for result in results]
df['score'] = [result['scores'][0] for result in results]

# write to csv. to be concatenated w/ other classifiedtweets
df.to_csv("/russeltweets_zeroshot_classified_deblarge_oct2424.csv")