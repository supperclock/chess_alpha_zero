from flask import render_template, jsonify, request
from flask_socketio import SocketIO, emit
from app import app
from app.engine.pikafish import PikafishEngine
import json

socketio = SocketIO(app)
engine = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global engine
    if not engine:
        engine = PikafishEngine('./pikafish')
        engine.start()
        
        def on_bestmove(move, ponder):
            emit('bestmove', {'move': move, 'ponder': ponder})
            
        def on_info(info):
            emit('engine_info', info)
            
        engine.add_listener('bestmove', on_bestmove)
        engine.add_listener('info', on_info)

@socketio.on('disconnect')
def handle_disconnect():
    global engine
    if engine:
        engine.stop()
        engine = None

@socketio.on('move')
def handle_move(data):
    if engine:
        engine.set_position(data['fen'], data.get('moves', []))
        engine.start_search({
            'wtime': data.get('wtime', 30000),
            'btime': data.get('btime', 30000),
            'winc': data.get('winc', 1000),
            'binc': data.get('binc', 1000)
        })

@socketio.on('stop')
def handle_stop():
    if engine:
        engine.stop_search()