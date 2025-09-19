import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from data_processing import (
    load_metadata,
    clean_metadata,
    publications_by_year,
    top_journals,
    most_common_title_words
)

st.set_page_config(page_title='CORD-19 Data Explorer', layout='wide')

st.title('CORD-19 Data Explorer')
st.write('A simple exploration of the CORD-19 metadata (prepared by Boaz Khuteka).')

@st.cache_data
def load_and_clean(path='metadata.csv', nrows=None):
    df = load_metadata(path, nrows=nrows)
    return clean_metadata(df)

uploaded = st.file_uploader('Upload metadata.csv', type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded, dtype=str)
    df = clean_metadata(df)
else:
    try:
        df = load_and_clean('metadata.csv', nrows=None)
    except FileNotFoundError:
        st.error('metadata.csv not found. Please upload it.')
        st.stop()

st.sidebar.header('Filters')
min_year = int(df['year'].dropna().min()) if df['year'].dropna().size else 2018
max_year = int(df['year'].dropna().max()) if df['year'].dropna().size else 2025
year_range = st.sidebar.slider('Select year range', min_year, max_year, (max_year-2, max_year))

df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]

st.header('Key metrics')
col1, col2, col3 = st.columns(3)
col1.metric('Total papers', len(df_filtered))
col2.metric('Unique journals', df_filtered['journal'].nunique())
col3.metric('Avg abstract words', int(df_filtered['abstract_word_count'].mean()))

st.subheader('Publications by year')
pubs = publications_by_year(df_filtered)
fig, ax = plt.subplots()
ax.bar(pubs.index.astype(int), pubs.values)
st.pyplot(fig)

st.subheader('Top journals')
journals = top_journals(df_filtered, top_n=15)
fig2, ax2 = plt.subplots()
journals.sort_values().plot(kind='barh', ax=ax2)
st.pyplot(fig2)

st.subheader('Common title words')
words = most_common_title_words(df_filtered, top_n=100)
text = ' '.join([(w + ' ') * count for w, count in words])
if text.strip():
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig3, ax3 = plt.subplots()
    ax3.imshow(wc, interpolation='bilinear')
    ax3.axis('off')
    st.pyplot(fig3)

st.subheader('Sample data')
st.dataframe(df_filtered[['title', 'journal', 'publish_time', 'year', 'source']].head(20))
