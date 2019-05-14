import os
from flask_socketio import send, emit
from flask_socketio import SocketIO
from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from flask import send_from_directory

import generate

app = Flask(__name__, static_url_path='')
app.logger.info('Loading model')
socketio = SocketIO(app)
enc, model = generate.init()
app.logger.info('Loaded model')

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
def generate_():
    context = request.get_json(force=True)
    phrase = context.get('text')
    generated = generate.main(model, enc, phrase)
    return jsonify({"response": "".join(generated)})


@socketio.on('client_connected')
def handle_client_connect_event(json):
    app.logger.info('received json: {0}'.format(str(json)))
    for text in "hi i am a robot".split(' '):
        app.logger.info('emitting... %s', text)
        emit('token', text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080), debug=False)
