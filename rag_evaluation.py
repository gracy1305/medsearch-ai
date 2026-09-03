from sentence_transformers import SentenceTransformer,util
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def parse_results_from_text(filename="/Users/kritikabhat/Downloads/ProjectMed 2/rag_results.txt"):
    with open(filename,"r",encoding="utf-8") as f:
        content = f.read()

    separator = "*-**-*"
    blocks = [b.strip() for b in content.split(separator) if "Query:" in b ]

    results = []
    for block in blocks:
        result = {}
        lines = block.split("\n")

        # Extract fields
        answer_lines = []
        in_ans = False

        for line in lines:
            if line.startswith("Query:"):
                result["Query"] = line.replace("Query:","").strip()
                in_ans = False
            elif line.startswith("Category:"):
                result["Category"] = line.replace("Category:","").strip()
                in_ans = False
            elif line.startswith("Retrieved:"):
                result["Retrieved"] = line.replace("Retrieved:","").strip()
                in_ans = False
            elif line.startswith("Answer:"):
                answer_lines = [line.replace("Answer:", "").strip()]
                in_ans = True
            elif line.startswith("Correct drug in answer:"):
                result["drug_mentioned_in_answer"] = "True" in line
                in_ans = False
            elif in_ans:
                answer_lines.append(line)

        result["answer"] = "\n".join(answer_lines).strip()
        if "Query" in result:
            results.append(result)

    return (results)

# sbert faithfulness
#compares ans to retrieved context text

def sbert_faithfulness(answer,retrieved_drugs):
    # higher cosine similarity means ans is more faithful retrieved docs
    emb = sbert_model.encode([answer,retrieved_drugs])
    score = util.cos_sim(emb[0],emb[1]).item()
    #  scale 0-1 to 1-3
    if score >= 0.6:
        return 3, round(score,3)
    elif score >= 0.35:
        return 2, round(score,3)
    else:
        return 1, round(score,3)
    
# manual relevance scoring
def manual_relevance_score(results):
    print("\n" + "=" * 60)
    print("Manual Relevance Score")
    print("Does the answer actually address the question?")
    print("  1 = Irrelevant: Does not answer the question")
    print("  2 = Partial: Incomplete or off-topic answer")
    print("  3 = Fully: Directly and completely answers")
    print("=" * 60 + "\n")

    for i , r in enumerate(results,start=1):
        print(f"Query: {i} / {len(results)}: {r['Query']}")
        print(f"Answer: {r['answer']}\n")

        while True:
            try:
                relevance = int(input("Relevance (1-3): "))
                if relevance in [1,2,3]:
                    break
                print("Please enter 1, 2, or 3.")
            except ValueError:
                print("Please enter a number")

        r['relevance'] = relevance
        print(f"Scored\n" + "-" * 60 + "\n")
    
    return results



# summary

def print_summary(results):
    n = len(results)
    avg_faith  = sum(r["faithfulness_score"] for r in results) / n
    avg_rel    = sum(r["relevance"]          for r in results) / n
    drug_correct = sum(1 for r in results if r["drug_mentioned_in_answer"])
 
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print(f"Drug mentioned in answer : {drug_correct}/{n} ({drug_correct/n*100:.1f}%)")
    print(f"Avg Faithfulness (SBERT) : {avg_faith:.2f} / 3")
    print(f"Avg Relevance (Manual)   : {avg_rel:.2f} / 3")
    print("=" * 60)

def save_results(results, filename="rag_scores.txt"):
    n = len(results)
    avg_faith  = sum(r["faithfulness_score"] for r in results) / n
    avg_rel    = sum(r["relevance"]          for r in results) / n
    drug_correct = sum(1 for r in results if r["drug_mentioned_in_answer"])
 
    with open(filename, "w", encoding="utf-8") as f:
        f.write("RAG EVALUATION RESULTS\n")
        f.write("Faithfulness: SBERT (answer vs retrieved context)\n")
        f.write("Relevance:    Manual scoring\n")
        f.write("=" * 60 + "\n\n")
 
        for i, r in enumerate(results, 1):
            f.write(f"Query {i}:      {r['Query']}\n")
            f.write(f"Category:     {r['Category']}\n")
            f.write(f"Retrieved:    {r['Retrieved']}\n")
            f.write(f"Answer:       {r['answer']}\n")
            f.write(f"Drug Correct: {r['drug_mentioned_in_answer']}\n")
            f.write(f"Faithfulness: {r['faithfulness_score']}/3 (SBERT cosine: {r['sbert_cosine']})\n")
            f.write(f"Relevance:    {r['relevance']}/3 (manual)\n")
            f.write("-" * 60 + "\n\n")
 
        f.write("SUMMARY\n")
        f.write(f"Drug mentioned in answer : {drug_correct}/{n} ({drug_correct/n*100:.1f}%)\n")
        f.write(f"Avg Faithfulness (SBERT) : {avg_faith:.2f} / 3\n")
        f.write(f"Avg Relevance (Manual)   : {avg_rel:.2f} / 3\n")
 
    print(f"\nResults saved to {filename}")
 
if __name__ == "__main__":
    print("Loading rag_results.txt...")
    results = parse_results_from_text("rag_results.txt")
    print(f"Loaded {len(results)} results\n")
 
    print("Computing SBERT faithfulness scores...")
    for r in results:
        score, cosine = sbert_faithfulness(r["answer"], r["Retrieved"])
        r["faithfulness_score"] = score
        r["sbert_cosine"] = cosine
    print("SBERT faithfulness done\n")
 
    results = manual_relevance_score(results)
    print_summary(results)
    save_results(results)

