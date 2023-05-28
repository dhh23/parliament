
# DHH 23 Parliament
# Code to add sentiment analysis to the data
# This will probably require a GPU to run
# This will add two columns, label and score, to the data
# The label is the sentiment, positive or negative
# The score is the confidence of the model in the label
# Author: Pontus Hedlund
# Date: 2023-05-28

import pandas as pd
from transformers import pipeline

data = pd.read_feather('../data/ParlaMint-GB-commons.feather')
speeches = data['speech'].tolist()

# Sentiment analysis
classifier = pipeline("sentiment-analysis", device=0)
sentiment_data = []
for out in classifier(speeches, batch_size=8, truncation=True):
    sentiment_data.append(out)

# Convert sentiment data to a dataframe
sentiment_df = pd.DataFrame(sentiment_data)

# Concate the sentiment data with the original data
new_df = pd.concat([data, sentiment_df], axis=1)

# Save the data
new_df.to_feather('../data/ParlaMint-GB-commons-with-sentiment.feather')
