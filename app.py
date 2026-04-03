# app.py
from flask import Flask, render_template, request
from src.chatbot import get_qa_chain

app = Flask(__name__)

# Initialize RAG QA chain once
qa = get_qa_chain(k=3)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "No input received!"
    
    result = qa({"query": msg})
    return str(result["result"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)