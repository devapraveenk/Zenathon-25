# app.py
from flask import Flask, request, jsonify
from crm_crud import CRMGraph

app = Flask(__name__)
crm_graph = CRMGraph()


@app.route("/process", methods=["POST"])
def process():
    """Endpoint to process user queries."""
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Process the query using the CRMGraph
    state = crm_graph.process_query(query)
    return jsonify(state)


if __name__ == "__main__":
    app.run(debug=True)
