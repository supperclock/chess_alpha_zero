"""
Flask backend for Xiangqi AlphaZero.

Endpoints (JSON):
- POST /api/new_game {human_side}
- GET  /api/state
- POST /api/move {from: [r,c], to: [r,c]}
- POST /api/ai_move
- GET  /api/settings
- POST /api/settings {sims, device, model_path}

Serves static frontend at '/'
"""

import os
import threading
from typing import Any, Dict, Tuple

from flask import Flask, jsonify, request, send_from_directory

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None  # type: ignore

from alphazero_xiangqi import (
    initial_board,
    generate_legal_moves,
    apply_move,
    is_terminal,
    AlphaNet,
    MCTS,
)


class GameManager:
    def __init__(self, model_path: str | None = None):
        self.lock = threading.Lock()
        self.board = initial_board()
        self.side_to_move = 'red'
        self.human_side = 'red'
        self.ai_thinking = False

        # engine
        self.device = 'cuda' if TORCH_OK and torch.cuda.is_available() else 'cpu'
        self.sims = 100
        self.model_path = model_path or os.environ.get('ALPHAXIANGQI_MODEL_PATH', 'alphazero_xiangqi.pt')
        self.net = None
        self.mcts = None
        self._ensure_engine(recreate_net=True, load_only=False)

    def _ensure_engine(self, recreate_net: bool = False, load_only: bool = False):
        if not TORCH_OK:
            self.net = None
        else:
            if recreate_net or self.net is None:
                self.net = AlphaNet()
            try:
                self.net.to(self.device)
            except Exception:
                self.device = 'cpu'
                self.net.to(self.device)
            if self.model_path and os.path.exists(self.model_path):
                try:
                    state = torch.load(self.model_path, map_location=self.device)
                    self.net.load_state_dict(state)
                except Exception:
                    pass
            else:
                if load_only:
                    pass
        self.mcts = MCTS(net=self.net, sims=int(self.sims), c_puct=1.0, device=self.device)

    def new_game(self, human_side: str):
        with self.lock:
            self.board = initial_board()
            self.side_to_move = 'red'
            self.human_side = 'red' if human_side not in ('red', 'black') else human_side
            self.ai_thinking = False

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            term, z = is_terminal(self.board)
            legal = generate_legal_moves(self.board, self.side_to_move)
            return {
                'board': self.board,
                'side_to_move': self.side_to_move,
                'human_side': self.human_side,
                'legal_moves': legal,
                'terminal': term,
                'result': z,
                'ai_thinking': self.ai_thinking,
            }

    def apply_human_move(self, move: Tuple[Tuple[int, int], Tuple[int, int]]):
        with self.lock:
            if self.side_to_move != self.human_side:
                raise ValueError('Not human\'s turn')
            legal = generate_legal_moves(self.board, self.side_to_move)
            if move not in legal:
                raise ValueError('Illegal move')
            self.board, _ = apply_move(self.board, move)
            self.side_to_move = 'red' if self.side_to_move == 'black' else 'black'

    def ai_move(self) -> Dict[str, Any]:
        with self.lock:
            if self.side_to_move == self.human_side:
                raise ValueError('Not AI\'s turn')
            if self.ai_thinking:
                raise ValueError('AI already thinking')
            self.ai_thinking = True

        try:
            move, _ = self.mcts.select_move(self.board, self.side_to_move, temperature=1e-3)
        except Exception as e:
            with self.lock:
                self.ai_thinking = False
            raise e

        with self.lock:
            if move is None:
                # no legal move
                term, z = True, (-1.0 if self.side_to_move == 'red' else 1.0)
                self.ai_thinking = False
                return {'move': None, 'terminal': term, 'result': z}
            self.board, _ = apply_move(self.board, move)
            self.side_to_move = 'red' if self.side_to_move == 'black' else 'black'
            term, z = is_terminal(self.board)
            self.ai_thinking = False
            return {'move': move, 'terminal': term, 'result': z}

    def update_settings(self, sims: int | None, device: str | None, model_path: str | None):
        changed_net = False
        if sims is not None:
            self.sims = int(sims)
        if device in ('cpu', 'cuda') and device != self.device:
            self.device = device
            changed_net = True
        if model_path is not None and model_path != self.model_path:
            self.model_path = model_path
            changed_net = True
        self._ensure_engine(recreate_net=changed_net, load_only=True)


app = Flask(__name__, static_folder='static', static_url_path='/static')
game = GameManager()


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def api_new_game():
    data = request.get_json(silent=True) or {}
    human_side = data.get('human_side', 'red')
    game.new_game(human_side)
    return jsonify({'ok': True, **game.get_state()})


@app.route('/api/state', methods=['GET'])
def api_state():
    return jsonify(game.get_state())


@app.route('/api/move', methods=['POST'])
def api_move():
    data = request.get_json(silent=True) or {}
    frm = data.get('from')
    to = data.get('to')
    if not (isinstance(frm, list) and isinstance(to, list) and len(frm) == 2 and len(to) == 2):
        return jsonify({'ok': False, 'error': 'Invalid move format'}), 400
    move = ((int(frm[0]), int(frm[1])), (int(to[0]), int(to[1])))
    try:
        game.apply_human_move(move)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, **game.get_state()})


@app.route('/api/ai_move', methods=['POST'])
def api_ai_move():
    try:
        result = game.ai_move()
    except Exception as e:
        print(e)
        return jsonify({'ok': False, 'error': str(e)}), 400
    state = game.get_state()
    return jsonify({'ok': True, 'ai': result, **state})


@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        return jsonify({'sims': game.sims, 'device': game.device, 'model_path': game.model_path, 'torch_available': TORCH_OK})
    data = request.get_json(silent=True) or {}
    sims = data.get('sims')
    device = data.get('device')
    model_path = data.get('model_path')
    try:
        game.update_settings(sims=sims, device=device, model_path=model_path)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 400
    return jsonify({'ok': True, **game.get_state()})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    if args.model_path:
        game.update_settings(sims=None, device=None, model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()


