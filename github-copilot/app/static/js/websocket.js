class ChessGame {
    constructor() {
        this.socket = io();
        this.board = new ChessBoard();
        this.isThinking = false;
        
        this.initSocketListeners();
        this.initUIControls();
    }
    
    initSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });
        
        this.socket.on('bestmove', (data) => {
            this.onBestMove(data);
        });
        
        this.socket.on('engine_info', (data) => {
            this.updateEngineInfo(data);
        });
    }
    
    initUIControls() {
        document.getElementById('newGame').addEventListener('click', () => {
            this.board.reset();
            this.notifyEngineNewPosition();
        });
        
        document.getElementById('flipBoard').addEventListener('click', () => {
            this.board.flipBoard();
        });
        
        this.board.onMove = (fen, moves) => {
            this.notifyEngineNewPosition(fen, moves);
        };
    }
    
    notifyEngineNewPosition(fen, moves) {
        this.isThinking = true;
        document.getElementById('thinking').style.display = 'block';
        
        this.socket.emit('move', {
            fen: fen,
            moves: moves,
            wtime: 30000,
            btime: 30000,
            winc: 1000,
            binc: 1000
        });
    }
    
    onBestMove(data) {
        this.isThinking = false;
        document.getElementById('thinking').style.display = 'none';
        
        if (data.move) {
            this.board.makeMove(data.move);
        }
    }
    
    updateEngineInfo(info) {
        const panel = document.getElementById('engine-info');
        let html = '';
        
        if (info.depth) {
            html += `深度: ${info.depth} `;
        }
        if (info.score) {
            const score = info.score.type === 'cp' ? 
                info.score.value / 100 : 
                `M${info.score.value}`;
            html += `分数: ${score} `;
        }
        if (info.nodes) {
            html += `节点数: ${info.nodes.toLocaleString()} `;
        }
        if (info.nps) {
            html += `NPS: ${info.nps.toLocaleString()}`;
        }
        
        panel.innerHTML = html;
    }
}

// 启动游戏
window.onload = () => {
    new ChessGame();
};