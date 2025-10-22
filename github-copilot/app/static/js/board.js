class ChessBoard {
    constructor() {
        this.board = document.getElementById('chessboard');
        this.selectedPiece = null;
        this.pieces = [];
        this.isFlipped = false;
        this.moves = [];
        
        // 棋子字符映射
        this.pieceChars = {
            'R': {'text': '车', 'class': 'red'},
            'N': {'text': '马', 'class': 'red'},
            'B': {'text': '相', 'class': 'red'},
            'A': {'text': '仕', 'class': 'red'},
            'K': {'text': '帅', 'class': 'red'},
            'C': {'text': '炮', 'class': 'red'},
            'P': {'text': '兵', 'class': 'red'},
            'r': {'text': '车', 'class': 'black'},
            'n': {'text': '马', 'class': 'black'},
            'b': {'text': '象', 'class': 'black'},
            'a': {'text': '士', 'class': 'black'},
            'k': {'text': '将', 'class': 'black'},
            'c': {'text': '炮', 'class': 'black'},
            'p': {'text': '卒', 'class': 'black'}
        };
        
        this.initBoard();
        this.initEventListeners();
    }
    
    initBoard() {
        this.drawGrid();
        this.initPieces();
    }
    
    drawGrid() {
        // 绘制横线
        for (let i = 0; i < 10; i++) {
            const line = document.createElement('div');
            line.className = 'grid-line horizontal-line';
            line.style.top = `${i * 100}px`;
            this.board.appendChild(line);
        }
        
        // 绘制竖线
        for (let i = 0; i < 9; i++) {
            const line = document.createElement('div');
            line.className = 'grid-line vertical-line';
            line.style.left = `${i * 100}px`;
            this.board.appendChild(line);
        }
    }
    
    createPiece(type, position) {
        const piece = document.createElement('div');
        const pieceInfo = this.pieceChars[type];
        
        piece.className = `piece ${pieceInfo.class}`;
        piece.dataset.type = type;
        piece.dataset.position = position;
        piece.dataset.text = pieceInfo.text;
        
        this.setPiecePosition(piece, position);
        this.board.appendChild(piece);
        this.pieces.push(piece);
        
        piece.addEventListener('click', () => this.onPieceClick(piece));
    }
    
    initPieces() {
        // 初始化所有棋子的位置
        const initialPosition = {
            'R': ['a0', 'i0'], 'N': ['b0', 'h0'], 'B': ['c0', 'g0'],
            'A': ['d0', 'f0'], 'K': ['e0'],
            'C': ['b2', 'h2'], 'P': ['a3', 'c3', 'e3', 'g3', 'i3'],
            'r': ['a9', 'i9'], 'n': ['b9', 'h9'], 'b': ['c9', 'g9'],
            'a': ['d9', 'f9'], 'k': ['e9'],
            'c': ['b7', 'h7'], 'p': ['a6', 'c6', 'e6', 'g6', 'i6']
        };
        
        for (let [type, positions] of Object.entries(initialPosition)) {
            for (let pos of positions) {
                this.createPiece(type, pos);
            }
        }
    }
    
    setPiecePosition(piece, position) {
        const [file, rank] = position.split('');
        const x = file.charCodeAt(0) - 'a'.charCodeAt(0);
        const y = 9 - parseInt(rank);
        
        const actualX = this.isFlipped ? 8 - x : x;
        const actualY = this.isFlipped ? 9 - y : y;
        
        piece.style.left = `${actualX * 100 + 10}px`;  // 居中调整
        piece.style.top = `${actualY * 100 + 10}px`;   // 居中调整
    }
    
    onPieceClick(piece) {
        if (this.selectedPiece === piece) {
            this.deselectPiece();
        } else if (!this.selectedPiece) {
            this.selectPiece(piece);
        } else {
            this.tryMove(piece);
        }
    }
    
    selectPiece(piece) {
        this.selectedPiece = piece;
        piece.classList.add('selected');
    }
    
    deselectPiece() {
        if (this.selectedPiece) {
            this.selectedPiece.classList.remove('selected');
            this.selectedPiece = null;
        }
    }
    
    tryMove(targetPiece) {
        const from = this.selectedPiece.dataset.position;
        const to = targetPiece.dataset.position;
        
        // 这里应该添加走子规则验证
        const move = `${from}${to}`;
        this.makeMove(move);
    }
    
    makeMove(move) {
        const [from, to] = [move.slice(0, 2), move.slice(2, 4)];
        
        // 更新棋子位置
        const piece = this.pieces.find(p => p.dataset.position === from);
        if (piece) {
            const targetPiece = this.pieces.find(p => p.dataset.position === to);
            if (targetPiece) {
                this.board.removeChild(targetPiece);
                this.pieces = this.pieces.filter(p => p !== targetPiece);
            }
            
            piece.dataset.position = to;
            this.setPiecePosition(piece, to);
        }
        
        this.moves.push(move);
        this.deselectPiece();
        
        // 通知引擎
        this.onMove(this.getCurrentFen(), this.moves);
    }
    
    getCurrentFen() {
        // 生成当前局面的FEN串
        return 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w';
    }
    
    flipBoard() {
        this.isFlipped = !this.isFlipped;
        for (let piece of this.pieces) {
            this.setPiecePosition(piece, piece.dataset.position);
        }
    }
    
    reset() {
        this.pieces.forEach(piece => this.board.removeChild(piece));
        this.pieces = [];
        this.moves = [];
        this.deselectPiece();
        this.initPieces();
    }
}