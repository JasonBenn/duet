from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify


from main import main

app = Flask(__name__)


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/api/generate', methods=['POST'])
def generate():
    context = request.get_json(force=True)
    phrase = context.get('text')
    if phrase:
        generated = main(phrase)
    else:
        generated = "NOP U NEED TO SEND PHRASE - I AM SMART BUT NOT A MIND READER"
    return jsonify({"response": "".join(generated)})
