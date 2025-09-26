"""
Human vs Neural Network Xiangqi UI using Tkinter.

Features:
- Loads trained parameters from alphazero_xiangqi.pt (or ALPHAXIANGQI_MODEL_PATH or --model_path)
- Select human side (red/black), start new game
- Configure MCTS simulations and device (cpu/cuda)
- Click-to-move with legality checking; AI replies using MCTS+NN
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None  # type: ignore

from alphazero_xiangqi import (
    initial_board,
    print_board,
    generate_legal_moves,
    apply_move,
    is_terminal,
    board_to_tensor,
    AlphaNet,
    MCTS,
)


PIECE_TO_GLYPH = {
    'r': '車', 'h': '馬', 'c': '炮', 'e': '象', 'a': '士', 'g': '將', 's': '卒',
    'R': '車', 'H': '馬', 'C': '炮', 'E': '相', 'A': '仕', 'G': '帥', 'S': '兵',
    '.': ''
}


class XiangqiUI:
    def __init__(self, root, model_path: str | None = None):
        self.root = root
        self.root.title("Xiangqi: Human vs AlphaZero-NN")

        # Game state
        self.board = initial_board()
        self.side_to_move = 'red'
        self.selected_square = None  # type: tuple[int,int] | None
        self.human_side = tk.StringVar(value='red')
        self.ai_thinking = False

        # Engine
        self.device = tk.StringVar(value=self._default_device())
        self.sims_var = tk.IntVar(value=100)
        self.model_path = tk.StringVar(value=model_path or os.environ.get('ALPHAXIANGQI_MODEL_PATH', 'alphazero_xiangqi.pt'))
        self.net = None
        self.mcts = None

        self._build_widgets()
        self._build_canvas()
        self._ensure_engine()
        self._redraw()

        # If human is black, let AI play first
        self.root.after(200, self._maybe_ai_move)

    def _default_device(self) -> str:
        if TORCH_OK and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _build_widgets(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="Human side:").pack(side=tk.LEFT, padx=4)
        side_cb = ttk.Combobox(toolbar, textvariable=self.human_side, values=['red','black'], state='readonly', width=6)
        side_cb.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Sims:").pack(side=tk.LEFT, padx=8)
        sims_spin = ttk.Spinbox(toolbar, from_=10, to=2000, increment=10, textvariable=self.sims_var, width=6)
        sims_spin.pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Device:").pack(side=tk.LEFT, padx=8)
        device_cb = ttk.Combobox(toolbar, textvariable=self.device, values=['cpu','cuda'], state='readonly', width=6)
        device_cb.pack(side=tk.LEFT)
        device_cb.bind('<<ComboboxSelected>>', lambda e: self._on_change_device())

        ttk.Button(toolbar, text="New Game", command=self._new_game).pack(side=tk.LEFT, padx=8)

        ttk.Label(toolbar, text="Model:").pack(side=tk.LEFT, padx=8)
        model_entry = ttk.Entry(toolbar, textvariable=self.model_path, width=30)
        model_entry.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Load", command=self._load_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Browse...", command=self._browse_model).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.root, textvariable=self.status_var, anchor='w')
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_canvas(self):
        self.canvas = tk.Canvas(self.root, width=540, height=600, bg="#f2e6c9", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        self.canvas.bind('<Button-1>', self._on_click)

    def _square_bbox(self, r: int, c: int):
        # Board 9 cols x 10 rows. Add margins.
        margin_x, margin_y = 30, 30
        cell_w = (self.canvas.winfo_width() or 540 - margin_x*2) / 9
        cell_h = (self.canvas.winfo_height() or 600 - margin_y*2) / 10
        x0 = margin_x + c * cell_w
        y0 = margin_y + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        return x0, y0, x1, y1

    def _redraw(self):
        self.canvas.delete('all')
        # grid
        margin_x, margin_y = 30, 30
        width = self.canvas.winfo_width() or 540
        height = self.canvas.winfo_height() or 600
        cell_w = (width - margin_x*2) / 9
        cell_h = (height - margin_y*2) / 10
        # horizontal lines
        for r in range(10):
            y = margin_y + r * cell_h
            self.canvas.create_line(margin_x, y, width - margin_x, y)
        # vertical lines
        for c in range(9):
            x = margin_x + c * cell_w
            self.canvas.create_line(x, margin_y, x, height - margin_y)

        # River text
        self.canvas.create_text(width/2, margin_y + 5*cell_h - cell_h/2, text="楚河    汉界", font=("Arial", 18))

        # pieces
        for r in range(10):
            for c in range(9):
                p = self.board[r][c]
                x0, y0, x1, y1 = self._square_bbox(r, c)
                fill = '#ffffff' if p != '.' else ''
                outline = '#333333' if p != '.' else ''
                if p != '.':
                    self.canvas.create_oval(x0+4, y0+4, x1-4, y1-4, fill=fill, outline=outline, width=2)
                    color = '#c00000' if p.isupper() else '#000000'
                    self.canvas.create_text((x0+x1)/2, (y0+y1)/2, text=PIECE_TO_GLYPH.get(p, p), fill=color, font=("Arial", 18, 'bold'))

        # selected highlight
        if self.selected_square is not None:
            r, c = self.selected_square
            x0, y0, x1, y1 = self._square_bbox(r, c)
            self.canvas.create_rectangle(x0+2, y0+2, x1-2, y1-2, outline="#00aaff", width=3)

    def _new_game(self):
        if self.ai_thinking:
            messagebox.showinfo("Info", "Please wait for AI to finish.")
            return
        self.board = initial_board()
        self.side_to_move = 'red'
        self.selected_square = None
        self.status_var.set("New game started.")
        self._redraw()
        self.root.after(200, self._maybe_ai_move)

    def _on_change_device(self):
        self._ensure_engine(recreate_net=True)

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Select model (.pt)", filetypes=[("PyTorch state_dict","*.pt"), ("All","*.*")])
        if path:
            self.model_path.set(path)

    def _load_model(self):
        self._ensure_engine(recreate_net=True, load_only=True)

    def _ensure_engine(self, recreate_net: bool = False, load_only: bool = False):
        if not TORCH_OK:
            self.status_var.set("PyTorch not available. Running in random MCTS mode.")
            self.net = None
        else:
            device = self.device.get()
            if recreate_net or self.net is None:
                self.net = AlphaNet()
            try:
                self.net.to(device)
            except Exception:
                self.device.set('cpu')
                device = 'cpu'
                self.net.to(device)
            model_path = self.model_path.get()
            if os.path.exists(model_path):
                try:
                    state = torch.load(model_path, map_location=device)
                    self.net.load_state_dict(state)
                    self.status_var.set(f"Loaded model: {model_path}")
                except Exception as e:
                    self.status_var.set(f"Failed to load model: {e}")
            else:
                if load_only:
                    self.status_var.set("Model file not found; using randomly initialized network.")
        # MCTS instance (can be recreated when sims changes; cheap)
        sims = int(self.sims_var.get())
        device = self.device.get()
        self.mcts = MCTS(net=self.net, sims=sims, c_puct=1.0, device=device)

    def _on_click(self, event):
        if self.ai_thinking:
            return
        # map click to square
        margin_x, margin_y = 30, 30
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        cell_w = (width - margin_x*2) / 9
        cell_h = (height - margin_y*2) / 10
        c = int((event.x - margin_x) // cell_w)
        r = int((event.y - margin_y) // cell_h)
        if not (0 <= r < 10 and 0 <= c < 9):
            return
        # Only allow move when it's human's turn
        if self.side_to_move != self.human_side.get():
            return
        p = self.board[r][c]
        if self.selected_square is None:
            if p == '.':
                return
            if (self.human_side.get() == 'red' and not p.isupper()) or (self.human_side.get() == 'black' and not p.islower()):
                return
            self.selected_square = (r, c)
            self._redraw()
            return
        else:
            r0, c0 = self.selected_square
            candidate = ((r0, c0), (r, c))
            legal = generate_legal_moves(self.board, self.side_to_move)
            if candidate in legal:
                self.board, _ = apply_move(self.board, candidate)
                self.side_to_move = 'red' if self.side_to_move == 'black' else 'black'
                self.selected_square = None
                self._redraw()
                term, z = is_terminal(self.board)
                if term:
                    self._on_game_end(z)
                else:
                    self.root.after(100, self._maybe_ai_move)
            else:
                # Either select another piece or clear
                if p != '.' and ((self.human_side.get() == 'red' and p.isupper()) or (self.human_side.get() == 'black' and p.islower())):
                    self.selected_square = (r, c)
                else:
                    self.selected_square = None
                self._redraw()

    def _maybe_ai_move(self):
        if self.side_to_move == self.human_side.get():
            return
        self._ensure_engine()  # refresh mcts sims/dev
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        threading.Thread(target=self._ai_move_thread, daemon=True).start()

    def _ai_move_thread(self):
        try:
            move, _ = self.mcts.select_move(self.board, self.side_to_move, temperature=1e-3)
        except Exception as e:
            move = None
            err = str(e)
            self.root.after(0, lambda: self.status_var.set(f"AI error: {err}"))
        if move is None:
            # no legal move
            self.root.after(0, lambda: self._on_game_end(-1.0 if self.side_to_move == 'red' else 1.0))
        else:
            def apply_and_continue():
                self.board, _ = apply_move(self.board, move)
                self.side_to_move = 'red' if self.side_to_move == 'black' else 'black'
                self._redraw()
                term, z = is_terminal(self.board)
                if term:
                    self._on_game_end(z)
                else:
                    self.status_var.set("Your move.")
                self.ai_thinking = False
            self.root.after(0, apply_and_continue)

    def _on_game_end(self, z: float):
        # z is +1 if red wins, -1 if black wins
        if z == 0:
            msg = "Draw"
        elif z > 0:
            msg = "Red wins"
        else:
            msg = "Black wins"
        self.status_var.set(msg)
        messagebox.showinfo("Game Over", msg)
        self.ai_thinking = False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, help='Path to .pt state_dict to load')
    args = parser.parse_args()
    root = tk.Tk()
    app = XiangqiUI(root, model_path=args.model_path)
    root.bind('<Configure>', lambda e: app._redraw())
    root.mainloop()


if __name__ == '__main__':
    main()


