# query_set.py
import random

# a smaller query set to compare retrieval
def sample_balanced_queries(all_queries, random_seed=42):

    random.seed(random_seed)

    side_effect_queries = [q for q in all_queries if q["category"] == "side_effects"]
    interaction_queries = [q for q in all_queries if q["category"] == "drug_interactions"]
    dosage_queries = [q for q in all_queries if q["category"] == "dosage"]

    sampled_queries = []

    sampled_queries.extend(random.sample(side_effect_queries, min(7, len(side_effect_queries))))
    sampled_queries.extend(random.sample(interaction_queries, min(7, len(interaction_queries))))
    sampled_queries.extend(random.sample(dosage_queries, min(6, len(dosage_queries))))

    random.shuffle(sampled_queries)
    return sampled_queries

# master query set for all drugs (***btw this is available to use, but hard to evalaute so did not use..***)
def create_queries_for_all_drugs(drug_corpus, min_section_length=20):
    """
    Each drug get three query types - side effects, drug interactions, and dosage/administration.
    """
    all_queries = []

    for index, row in drug_corpus.iterrows():
        drug_name = str(row["drug_name"]).strip()

        side_effects_text = str(row["side_effects"]).strip()
        interactions_text = str(row["drug_interactions"]).strip()
        dosage_text = str(row["dosage_and_administration"]).strip()

        if len(drug_name) == 0:
            continue

        if len(side_effects_text) > min_section_length:
            all_queries.append({
                "query_id": f"D{index}_SIDE",
                "query": f"What are the side effects and adverse reactions of {drug_name}?",
                "relevant_drugs": [drug_name],
                "category": "side_effects",
                "target_section": "side_effects"
            })

        if len(interactions_text) > min_section_length:
            all_queries.append({
                "query_id": f"D{index}_INT",
                "query": f"What drug interactions are listed for {drug_name}?",
                "relevant_drugs": [drug_name],
                "category": "drug_interactions",
                "target_section": "drug_interactions"
            })

        if len(dosage_text) > min_section_length:
            all_queries.append({
                "query_id": f"D{index}_DOSE",
                "query": f"What is the dosage and administration for {drug_name}?",
                "relevant_drugs": [drug_name],
                "category": "dosage",
                "target_section": "dosage_and_administration"
            })

    return all_queries

if __name__ == "__main__":
    import json
    import pandas as pd

    with open("/Users/kritikabhat/Downloads/ProjectMed 2/ProjectMed/drug_corpus_json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    drug_corpus = pd.DataFrame(documents)

    all_queries = create_queries_for_all_drugs(drug_corpus)
    sampled = sample_balanced_queries(all_queries)

    print(f"Total queries generated: {len(all_queries)}")
    print(f"Sampled queries: {len(sampled)}")
    for q in sampled:
        print(q)