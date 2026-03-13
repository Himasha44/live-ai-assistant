from flask import Flask, request, jsonify
import json
import pickle
import random
import re
from datetime import datetime
import os

app = Flask(__name__)

with open("intents.json", "r", encoding="utf-8") as file:
    intents_data = json.load(file)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

MEMORY_FILE = "chat_memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file)

def predict_intent(user_input):
    X = vectorizer.transform([user_input])
    probabilities = model.predict_proba(X)[0]

    max_prob = max(probabilities)
    prediction = model.classes_[probabilities.argmax()]

    print("User input:", user_input)
    print("Predicted intent:", prediction)
    print("Confidence:", max_prob)

    if max_prob < 0.20:
        return "unknown"

    return prediction

def get_response(intent_tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Sorry, I do not understand."

def calculate_expression(user_input):
    expression = re.findall(r'[\d\+\-\*\/\(\)\. ]+', user_input)
    if expression:
        expr = "".join(expression).strip()
        if expr:
            try:
                result = eval(expr)
                return f"The answer is {result}"
            except Exception:
                return None
    return None

def store_name(user_input, memory):
    match = re.search(r"(my name is|i am|call me)\s+([A-Za-z]+)", user_input.lower())
    if match:
        name = match.group(2).capitalize()
        memory["name"] = name
        save_memory(memory)
        return f"Nice to meet you, {name}!"
    return None

def get_name(memory):
    if "name" in memory:
        return f"Your name is {memory['name']}."
    return "I do not know your name yet."

def check_rules(user_input, memory):
    lower_input = user_input.lower()

    name_response = store_name(user_input, memory)
    if name_response:
        return name_response

    if "what is my name" in lower_input or "tell me my name" in lower_input or "do you know my name" in lower_input:
        return get_name(memory)

    if "time" in lower_input:
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}"

    if "date" in lower_input or "today" in lower_input:
        return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}"

    calc_result = calculate_expression(user_input)
    if calc_result:
        return calc_result

    return None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        user_input = str(data.get("message", "")).strip()

        if not user_input:
            return jsonify({"answer": "Please type a message."}), 400

        memory = load_memory()

        rule_response = check_rules(user_input, memory)
        if rule_response:
            return jsonify({"answer": rule_response})

        intent = predict_intent(user_input)
        response = get_response(intent)

        return jsonify({"answer": response, "intent": intent})

    except Exception as e:
        print("Python error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)