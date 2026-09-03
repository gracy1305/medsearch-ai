import os
import ollama
from retrieval_system import search_with_bm25, drug_corpus
from query_set import create_queries_for_all_drugs,sample_balanced_queries

"""
BM25 retrieval → finds relevant drug documents.
Ollama (LLM) → generates an answer using only the retrieved information.
Evaluation framework → tests how well the RAG system works.
"""

def build_context(top_docs):
    context = ""
    for _,row in top_docs.iterrows():
        context += f"Drug: {row['drug_name']}\n"
        context += f"Side Effects: {row['side_effects']}\n"
        context += f"Interactions: {row['drug_interactions']}\n"
        context += f"Dosage: {row['dosage_and_administration']}\n"
        context += "-"* 40 + "\n"

    return context.strip()

def rag_answer(user_query,top_k = 3):
    # input would be like: "what are the side effects of aspirin"
    top_docs = search_with_bm25(user_query,top_k=top_k)
    context = build_context(top_docs)

    """
    Drug: Aspirin
    Side Effects: Nausea, bleeding
    ...
    """
    prompt = f"""You are a medical information assistant. Use only the context provided below to answer the question. 
    If the answer is not found in the context, say "I don't have enough information to answer that."
    Be concise and specific.

    context:
    {context}

    Question: {user_query}
    Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "query": user_query,
        "retrieved_drugs": top_docs["drug_name"].tolist(),
        "answer": response["message"]["content"]
    }

def evaluate_rag(sampled_queries,top_k=3):
    results = []
    total = len(sampled_queries)
    for i, query_item in enumerate(sampled_queries,start=1):
        print(f"Processing query {i}/{total}: {query_item['query'][:60]}...")
        output = rag_answer(query_item["query"],top_k=top_k)
        output["relevant_drugs"] = query_item["relevant_drugs"]
        output["category"] = query_item["category"]

        # Did the correct drug appear in the answer?
        correct_drug = query_item["relevant_drugs"][0].lower()
        output["drug_mentioned_in_answer"] = correct_drug in output["answer"].lower()

        results.append(output)
        print(f"✓ Done ({i}/{total})\n")
    return results

# def print_results(results):
#     correct = sum(1 for r in results if r["drug_mentioned_in_answer"])
#     print(f"\nRAG Evaluation - {len(results)} queries")
#     print(f"Drug mentioned in answer: {correct}/{len(results)}({correct/len(results)*100:.1f}%)\n")
#     print("*-*" * 50)
#     for r in results:
#         print(f"Query:     {r['query']}")
#         print(f"Retrieved: {r['retrieved_drugs']}")
#         print(f"Answer:    {r['answer']}")
#         print(f"Correct drug in answer: {r['drug_mentioned_in_answer']}")
#         print("*-*" * 50)

def save_results_file(results,filename="rag_results.txt"):
    correct = sum(1 for r in results if r["drug_mentioned_in_answer"])
    with open(filename,"w",encoding="utf-8") as f:
        f.write(f"\nRAG Evaluation - {len(results)} queries\n") 
        f.write(f"Drug mentioned in answer: {correct}/{len(results)}({correct/len(results)*100:.1f}%)\n")
        f.write("*-*" * 50 + "\n\n")
        for r in results:
            f.write(f"Query:     {r['query']}\n")
            f.write(f"Category:  {r['category']}\n")
            f.write(f"Retrieved: {r['retrieved_drugs']}\n")
            f.write(f"Answer:    {r['answer']}\n")
            f.write(f"Correct drug in answer: {r['drug_mentioned_in_answer']}\n")
            f.write("*-*" * 50 + "\n\n")

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    all_queries = create_queries_for_all_drugs(drug_corpus)
    sampled_queries = sample_balanced_queries(all_queries)

    results = evaluate_rag(sampled_queries)
    save_results_file(results)