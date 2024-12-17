import pandas as pd
import matplotlib.pyplot as plt

# verilleri birleştirip shufflelama
# df = pd.read_csv('temizlenmis_veri.csv',sep="|")
# df2 = pd.read_csv("non_offensive.csv",sep="|",encoding="utf-8")

# merged_df = pd.concat([df,df2],ignore_index=True)
# shuffled_df = merged_df.sample(frac=1,random_state=42).reset_index(drop=True)

# shuffled_df.to_csv("augmented_dataset.csv",sep="|",index=False)

# eksik verilerin silinmesi
# original_row_count = len(df)

# df_cleaned = df.dropna()

# dropped_rows = original_row_count - len(df_cleaned)
# print(f"Silinen satır sayısı: {dropped_rows}")

# df_cleaned.to_csv('temizlenmis_veri.csv', index=False,sep="|")

# dağılımın pasta grafiği
# is_offensive_counts = df['is_offensive'].value_counts(normalize=True) * 100
# target_counts = df['target'].value_counts(normalize=True) * 100


# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].pie(is_offensive_counts, labels=is_offensive_counts.index, autopct='%1.1f%%', startangle=90)
# ax[0].set_title('is_offensive Sütunu')


# ax[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
# ax[1].set_title('target Sütunu')
# plt.subplots_adjust(top=0.85)
# plt.tight_layout()
# plt.show()

# PREPROCESSING

# METİN TEMİZLEME 

# 1) Küçük Harf Dönüşümü ve Sembollerin Çıkartılması
import re

def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', '', text)

# 2) Stopwordsların Kaldırılması

def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = file.read().splitlines()
    return set(stopwords)

def remove_custom_stopwords(text, stopwords):
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

stopwords = load_stopwords('stopwords.txt')

text = "Araplar ve Çinliler tehlikelidir ama Japonlar değildir"

cleaned_text = remove_custom_stopwords(text, stopwords)
print(cleaned_text)

# 3) N-GRAM çıkarma
from sklearn.feature_extraction.text import CountVectorizer

def extract_ngrams(text, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

bigrams = extract_ngrams(text, 2)  # Bigrams (2-grams)
print(bigrams)

