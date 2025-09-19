"""
A Jupyter-friendly script for the Frameworks Assignment.
Author: Boaz Khuteka
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import (
    load_metadata,
    basic_explore,
    clean_metadata,
    publications_by_year,
    top_journals,
    most_common_title_words
)

print("Loading metadata...")
df = load_metadata('metadata.csv', nrows=50000)  # sample for speed
print('Shape:', df.shape)
print(df.columns.tolist())

# Basic exploration
info = basic_explore(df)
print('Missing values (first 10):')
for k, v in list(info['missing'].items())[:10]:
    print(k, v)

# Cleaning
clean = clean_metadata(df)
print('Years:', clean['year'].dropna().unique()[:10])

# Plot publications by year
pubs = publications_by_year(clean)
plt.bar(pubs.index.astype(int), pubs.values)
plt.title('Publications by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()

# Top journals
print("Top journals:")
print(top_journals(clean, top_n=10))

# Common title words
print("Common title words:")
print(most_common_title_words(clean, top_n=30))
