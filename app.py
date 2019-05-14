from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from flask import send_from_directory


from main import main

app = Flask(__name__, static_url_path='')


@app.route('/hello')
def hello_world():
    return 'Hello, World!'


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


@app.route('/api/generate', methods=['POST'])
def generate():
    context = request.get_json(force=True)
    phrase = context.get('text')
    if phrase:
        generated = main(phrase)
    else:
        generated = "NOP U NEED TO SEND PHRASE - I AM SMART BUT NOT A MIND READER"
    return jsonify({"response": "".join(generated)})

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
