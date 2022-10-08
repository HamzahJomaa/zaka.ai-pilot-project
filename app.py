from flask import Flask, request, jsonify
from Functions.preprocessing import clean_text
import os

from Classes.Model import Model

app = Flask(__name__)
app.testing = True

model = Model("Models/Tokenizers", "Models/Inference", max_len_en=9, max_len_fr=24)
model.load_files()


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    sequence = model.prepocess(data["text"])
    text = model.translate_to_english(sequence)
    return jsonify({"text":text})


@app.route("/")
def MainApp():
    sequence = model.prepocess("she is driving the truck")
    print(model.translate_to_english(sequence))
    return "<p>Zaka Endpoints</p>"



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)