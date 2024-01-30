import pandas as pd
import numpy as np
from scipy import stats
from packaging import version
import matplotlib.pyplot as plt
import seaborn as sns


def clean_text(text):
    # Remove special characters and numbers, convert to lowercase
    return text.str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()


df = pd.read_csv('data/Winter_2024_Scotia_DSD_Data_Set.csv')
df = df[df['Review_Language'] == 'en']


df['Review'] = clean_text(df['Review'])
missing_reviews = df['Review'].isnull().sum()
print(f'There are {missing_reviews} missing reviews')
df = df.dropna(subset=['Review'])

df = df.drop(['Review_Language', 'Date', 'Version',
             'Review_Likes', 'Rating'], axis=1)


df_train = df.sample(frac=0.05, random_state=0)
df_test = df.drop(df_train.index)
print(
    f'There are {len(df_train)} reviews in the training set and {len(df_test)} in the test set')

df_train.to_csv('data/training_data.csv', index=False)
