# evaluation.py
import math
import pandas as pd

def is_relevant_result(retrieved_drug_name, relevant_drugs):
    retrieved_name = str(retrieved_drug_name).lower().strip()

    for drug in relevant_drugs:
        expected_name = str(drug).lower().strip()
        if retrieved_name == expected_name:
            return 1
    return 0


def evaluate_retrieval(search_function, sampled_queries, k_values=[1, 3, 5, 10]):
    evaluation_results = []

    for k in k_values:
        correct_queries = 0
        precision_total = 0
        recall_total = 0
        ndcg_total = 0

        for query_item in sampled_queries:
            results = search_function(query_item["query"], top_k=k)
            relevant_drugs = query_item["relevant_drugs"]

            relevance_scores = [
                is_relevant_result(drug_name, relevant_drugs)
                for drug_name in results["drug_name"]]
            
            retrieved_relevant = int(sum(relevance_scores) > 0)
            total_relevant = len(relevant_drugs)

            # Hit rate/Top-K Accuracy
            if retrieved_relevant > 0:
                correct_queries += 1

            # Precision@k
            precision_total += retrieved_relevant / k

            # Recall@k
            recall_total += retrieved_relevant

            # nDCG
            dcg = 0
            for rank, relevance in enumerate(relevance_scores, start=1):
                dcg += relevance / math.log2(rank + 1)

            ideal_scores = sorted(relevance_scores, reverse=True)

            idcg = 0
            for rank, relevance in enumerate(ideal_scores, start=1):
                idcg += relevance / math.log2(rank + 1)
            if idcg > 0:
                ndcg_total += dcg / idcg

        num_queries = len(sampled_queries)

        evaluation_results.append({
            "k": k,
            "top_k_accuracy": correct_queries / num_queries,
            "precision_at_k": precision_total / num_queries,
            "recall_at_k": recall_total/ num_queries,
            "ndcg_at_k": ndcg_total / num_queries
        })
    return pd.DataFrame(evaluation_results)