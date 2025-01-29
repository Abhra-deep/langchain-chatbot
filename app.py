from flask import Flask, request, jsonify
from chatbot import find_relevant_chunk

# Initialize Flask app
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    response = find_relevant_chunk(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5005)
