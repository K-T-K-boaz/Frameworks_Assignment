
---

### 4️⃣ `data_processing.py`
Save this as **data_processing.py**
```python
"""
Helpers for loading and cleaning the CORD-19 metadata.csv file.
Author: Boaz Khuteka
"""

import pandas as pd
import numpy as np
from collections import Counter
import re


def load_metadata(path="metadata.csv", nrows=None):
    """Load metadata.csv into a DataFrame."""
    df = pd.read_csv(path, dtype=str, nrows=nrows)
    return df


def basic_explore(df):
    """Return basic exploration info as dict."""
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict(),
        'missing': df.isnull().sum().to_dict(),
    }
    return info


def clean_metadata(df):
    """Perform cleaning steps and return a cleaned copy."""
    df = df.copy()

    # publish_time -> datetime
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
        df['year'] = df['publish_time'].dt.year
    else:
        df['year'] = np.nan

    # Word counts
    df['title'] = df['title'].fillna('')
    df['abstract'] = df.get('abstract', pd.Series([''] * len(df))).fillna('')

    df['title_word_count'] = df['title'].apply(lambda s: len(str(s).split()))
    df['abstract_word_count'] = df['abstract'].apply(lambda s: len(str(s).split()))

    # Journal cleanup
    if 'journal' in df.columns:
        df['journal'] = df['journal'].fillna('Unknown').str.strip()

    # Source cleanup
    if 'source_x' in df.columns:
        df['source'] = df['source_x'].fillna(df.get('source_y', 'unknown'))
    elif 'source' not in df.columns:
        df['source'] = 'unknown'

    return df


def top_journals(df, top_n=20):
    if 'journal' not in df.columns:
        return []
    return df['journal'].value_counts().head(top_n)


def publications_by_year(df):
    if 'year' not in df.columns:
        return pd.Series(dtype=int)
    return df['year'].value_counts().sort_index()


def most_common_title_words(df, top_n=50, stopwords=None):
    stopwords = stopwords or {
        'the', 'and', 'of', 'in', 'a', 'to', 'for', 'on', 'with', 'by',
        'an', 'from', 'study', 'evidence', 'analysis', 'using'
    }
    words = Counter()
    for t in df['title'].dropna().astype(str):
        t = re.sub(r"[^a-zA-Z0-9\\s]", ' ', t).lower()
        for w in t.split():
            if w and w not in stopwords and len(w) > 2:
                words[w] += 1
    return words.most_common(top_n)
