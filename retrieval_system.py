import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np
from rank_bm25 import BM25Okapi
from query_set import create_queries_for_all_drugs, sample_balanced_queries
from evaluation import evaluate_retrieval
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "drug_corpus_json")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

drug_corpus = pd.DataFrame(documents)
drug_corpus = drug_corpus.drop_duplicates().reset_index(drop=True) # Removing exact duplicates

# Singular searchable text column for each drug doc
drug_corpus["search_text"] = (
    drug_corpus["drug_name"].fillna("") + " " +
    drug_corpus["side_effects"].fillna("") + " " +
    drug_corpus["drug_interactions"].fillna("") + " " +
    drug_corpus["dosage_and_administration"].fillna("")
)

drug_corpus["search_text"] = (
    drug_corpus["search_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(drug_corpus["search_text"])
print(tfidf_matrix.shape) #1468 documents after removing exact duplicates and 199,150 TF-IDF features

# Dense Retriever using LSA/SVD
svd_model = TruncatedSVD(n_components=100, random_state=42)
dense_doc_matrix = svd_model.fit_transform(tfidf_matrix)
dense_doc_matrix = normalize(dense_doc_matrix)

def search_with_dense(user_query, top_k=5):
    query_tfidf = tfidf_vectorizer.transform([user_query])
    query_dense = svd_model.transform(query_tfidf)
    query_dense = normalize(query_dense)

    dense_scores = cosine_similarity(query_dense, dense_doc_matrix).flatten()
    top_result_indices = dense_scores.argsort()[::-1][:top_k]

    top_results = drug_corpus.iloc[top_result_indices][[
        "drug_name",
        "side_effects",
        "drug_interactions",
        "dosage_and_administration"
    ]].copy()

    top_results["dense_score"] = dense_scores[top_result_indices]
    return top_results

all_queries = create_queries_for_all_drugs(drug_corpus)
sampled_queries = sample_balanced_queries(all_queries)

def search_with_tfidf(user_query, top_k=5):
    query_vector = tfidf_vectorizer.transform([user_query])  # Converts the user query into the TF-IDF format 
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_result_indices = similarity_scores.argsort()[::-1][:top_k]

    top_results = drug_corpus.iloc[top_result_indices][[
        "drug_name",
        "side_effects",
        "drug_interactions",
        "dosage_and_administration"
    ]].copy()

    top_results["tfidf_score"] = similarity_scores[top_result_indices]
    return top_results



# BM25

bm25_stopwords = {
    "what", "are", "is", "the", "and", "for", "of", "to",
    "listed", "drug", "drugs", "side", "effects", "adverse",
    "reactions", "interactions", "dosage", "administration"}


def tokenize_text(text):
    text = str(text).lower()
    tokens = re.findall(r"\b\w+\b", text)
    tokens = [token for token in tokens if token not in bm25_stopwords]
    return tokens


tokenized_corpus = drug_corpus["search_text"].apply(tokenize_text).tolist()
bm25_model = BM25Okapi(tokenized_corpus)

def search_with_bm25(user_query, top_k=5):
    tokenized_query = tokenize_text(user_query)
    bm25_scores = bm25_model.get_scores(tokenized_query)
    top_result_indices = bm25_scores.argsort()[::-1][:top_k]
    top_results = drug_corpus.iloc[top_result_indices][[
        "drug_name",
        "side_effects",
        "drug_interactions",
        "dosage_and_administration"
    ]].copy()

    top_results["bm25_score"] = bm25_scores[top_result_indices]
    return top_results

def normalize_scores(scores):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def search_with_hybrid(user_query, top_k=5, bm25_weight=0.6, dense_weight=0.4):
    tokenized_query = tokenize_text(user_query)
    bm25_scores = bm25_model.get_scores(tokenized_query)

    query_tfidf = tfidf_vectorizer.transform([user_query])
    query_dense = svd_model.transform(query_tfidf)
    query_dense = normalize(query_dense)
    dense_scores = cosine_similarity(query_dense, dense_doc_matrix).flatten()

    bm25_norm = normalize_scores(bm25_scores)
    dense_norm = normalize_scores(dense_scores)

    hybrid_scores = (bm25_weight * bm25_norm) + (dense_weight * dense_norm)

    top_result_indices = hybrid_scores.argsort()[::-1][:top_k]

    top_results = drug_corpus.iloc[top_result_indices][[
        "drug_name",
        "side_effects",
        "drug_interactions",
        "dosage_and_administration"
    ]].copy()

    top_results["hybrid_score"] = hybrid_scores[top_result_indices]
    return top_results

#Retriever Evaluation
tfidf_evaluation = evaluate_retrieval(search_with_tfidf, sampled_queries)
bm25_evaluation = evaluate_retrieval(search_with_bm25, sampled_queries)
dense_evaluation = evaluate_retrieval(search_with_dense, sampled_queries)
hybrid_evaluation = evaluate_retrieval(search_with_hybrid, sampled_queries)

print("\nTF-IDF Eval Results:")
print(tfidf_evaluation)

print("\nBM25 Eval Results:")
print(bm25_evaluation)

print("\nDense Eval Results:")
print(dense_evaluation)

print("\nHybrid Eval Results:")
print(hybrid_evaluation)