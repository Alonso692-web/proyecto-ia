from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Cargar el modelo y el tokenizador desde Hugging Face
model_name = 'alonso4/final-proyect'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Función para generar un poema a partir de un prompt dado
def generate_poem_with_model(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return poem

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    max_length = data.get('max_length', 30)  # Obtener max_length del cuerpo de la solicitud
    poem = generate_poem_with_model(prompt, max_length=max_length)
    return jsonify({'poem': poem})

if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    # debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # app.run(host='0.0.0.0', port=port, debug=debug)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"Attempting to run on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)