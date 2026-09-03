from flask import Flask, request, render_template_string
from retrieval_system import (
    search_with_tfidf,
    search_with_bm25,
    search_with_dense,
    search_with_hybrid
)

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MedSearch AI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f7f9fc; }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 12px; }
        h1 { color: #002b5c; }
        input, select { width: 100%; padding: 12px; margin: 10px 0 20px; font-size: 16px; }
        button { background: #0078D7; color: white; padding: 12px 22px; border: none; cursor: pointer; font-size: 16px; }
        .result { margin-top: 25px; padding: 18px; border-left: 5px solid #0078D7; background: #f1f7ff; }
        .section { margin-top: 10px; }
        .label { font-weight: bold; color: #002b5c; }
    </style>
</head>
<body>
<div class="container">
    <h1>MedSearch AI</h1>
    <p>FDA Drug Label Search System</p>

    <form method="POST">
        <label>Enter Query</label>
        <input type="text" name="query" placeholder="Example: Side effects of ibuprofen" value="{{ query }}">

        <label>Select Retriever</label>
        <select name="retriever">
            <option {% if retriever == 'BM25' %}selected{% endif %}>BM25</option>
            <option {% if retriever == 'TF-IDF' %}selected{% endif %}>TF-IDF</option>
            <option {% if retriever == 'Dense' %}selected{% endif %}>Dense</option>
            <option {% if retriever == 'Hybrid' %}selected{% endif %}>Hybrid</option>
        </select>

        <button type="submit">Search</button>
    </form>

    {% if results %}
        <h2>Results</h2>
        {% for row in results %}
            <div class="result">
                <h3>{{ row.drug_name }}</h3>

                <div class="section">
                    <span class="label">Side Effects:</span>
                    <p>{{ row.side_effects }}</p>
                </div>

                <div class="section">
                    <span class="label">Drug Interactions:</span>
                    <p>{{ row.drug_interactions }}</p>
                </div>

                <div class="section">
                    <span class="label">Dosage and Administration:</span>
                    <p>{{ row.dosage_and_administration }}</p>
                </div>
            </div>
        {% endfor %}
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    query = ""
    retriever = "BM25"

    if request.method == "POST":
        query = request.form.get("query", "")
        retriever = request.form.get("retriever", "BM25")

        if query.strip():
            if retriever == "TF-IDF":
                df = search_with_tfidf(query, top_k=3)
            elif retriever == "Dense":
                df = search_with_dense(query, top_k=3)
            elif retriever == "Hybrid":
                df = search_with_hybrid(query, top_k=3)
            else:
                df = search_with_bm25(query, top_k=3)

            results = df.to_dict(orient="records")

    return render_template_string(
        HTML,
        results=results,
        query=query,
        retriever=retriever
    )

if __name__ == "__main__":
    app.run(debug=True)