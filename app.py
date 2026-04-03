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

    # 🧹 Remove timestamps like 18:19 if ever present
    import re
    msg = re.sub(r'\d{1,2}:\d{2}', '', msg).strip()

    print("User Query:", msg)

    result = qa.invoke({"query": msg})
    return result["result"]


# 🔥 IMPORTANT: This was missing
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)