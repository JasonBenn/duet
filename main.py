import logging
import os

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask.logging import default_handler
from flask_socketio import SocketIO, emit, send

import generate
import generate_wiki

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('waitress')
logger.addHandler(default_handler)

app = Flask(__name__, static_url_path='')
app.logger.info('Loading model')
# socketio = SocketIO(app)
enc, model = generate.init()
try:
    wiki_model = generate_wiki.init()
except Exception as e:
    wiki_model = None
    app.logger.warning('Wiki model not found', e)
app.logger.info('Loaded model')

@app.route('/hello')
def hello_world():
    return 'Hello, World!'

@app.route('/readiness_check')
def ready():
    return 'OK'

@app.route('/liveness_check')
def live():
    return 'OK'

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


@app.route('/api/generate', methods=['POST'])
def generate_():
    context = request.get_json(force=True)
    phrase = context.get('text', '')
    if context.get('model') == 'wiki' and wiki_model:
        generated = generate_wiki.main(wiki_model, enc, phrase)
    else:
        generated = generate.main(model, enc, phrase)
    return jsonify({"response": generated})


# @socketio.on('client_connected')
# def handle_client_connect_event(json):
#     app.logger.info('received json: {0}'.format(str(json)))
#     for text in "hi i am a robot".split(' '):
#         app.logger.info('emitting... %s', text)
#         emit('token', text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080), debug=False)
