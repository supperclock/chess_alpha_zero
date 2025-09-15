class ChineseChess {
    constructor() {
        this.board = [];
        this.currentPlayer = 'red';
        this.gameStatus = 'playing';
        this.selectedPiece = null;
        this.validMoves = [];
        
        this.initializeBoard();
        this.setupEventListeners();
        this.updateGameInfo();
    }
    
    initializeBoard() {
        const chessboard = document.getElementById('chessboard');
        chessboard.innerHTML = '';
        
        // 添加九宫格斜线
        const palaceLines = document.createElement('div');
        palaceLines.className = 'palace-lines';
        palaceLines.innerHTML = `
            <div class="palace-line top-left"></div>
            <div class="palace-line top-right"></div>
            <div class="palace-line bottom-left"></div>
            <div class="palace-line bottom-right"></div>
        `;
        chessboard.appendChild(palaceLines);
        
        // 添加河界文字
        const riverTop = document.createElement('div');
        riverTop.className = 'river top';
        riverTop.textContent = '楚';
        chessboard.appendChild(riverTop);
        
        const riverBottom = document.createElement('div');
        riverBottom.className = 'river bottom';
        riverBottom.textContent = '河';
        chessboard.appendChild(riverBottom);
        
        // 创建棋盘格子
        for (let row = 0; row < 10; row++) {
            this.board[row] = [];
            for (let col = 0; col < 9; col++) {
                const square = document.createElement('div');
                square.className = 'square';
                square.dataset.row = row;
                square.dataset.col = col;
                
                square.addEventListener('click', () => this.handleSquareClick(row, col));
                
                chessboard.appendChild(square);
                this.board[row][col] = square;
            }
        }
        
        this.loadInitialPieces();
    }
    
    loadInitialPieces() {
        // 初始棋子布局
        const initialSetup = [
            // 黑方(上方)
            {row: 0, col: 0, type: 'chariot', color: 'black'},
            {row: 0, col: 1, type: 'horse', color: 'black'},
            {row: 0, col: 2, type: 'elephant', color: 'black'},
            {row: 0, col: 3, type: 'advisor', color: 'black'},
            {row: 0, col: 4, type: 'general', color: 'black'},
            {row: 0, col: 5, type: 'advisor', color: 'black'},
            {row: 0, col: 6, type: 'elephant', color: 'black'},
            {row: 0, col: 7, type: 'horse', color: 'black'},
            {row: 0, col: 8, type: 'chariot', color: 'black'},
            {row: 0, col: 1, type: 'cannon', color: 'black'},
            {row: 0, col: 7, type: 'cannon', color: 'black'},
            {row: 3, col: 0, type: 'soldier', color: 'black'},
            {row: 3, col: 2, type: 'soldier', color: 'black'},
            {row: 3, col: 4, type: 'soldier', color: 'black'},
            {row: 3, col: 6, type: 'soldier', color: 'black'},
            {row: 3, col: 8, type: 'soldier', color: 'black'},
            
            // 红方(下方)
            {row: 9, col: 0, type: 'chariot', color: 'red'},
            {row: 9, col: 1, type: 'horse', color: 'red'},
            {row: 9, col: 2, type: 'elephant', color: 'red'},
            {row: 9, col: 3, type: 'advisor', color: 'red'},
            {row: 9, col: 4, type: 'general', color: 'red'},
            {row: 9, col: 5, type: 'advisor', color: 'red'},
            {row: 9, col: 6, type: 'elephant', color: 'red'},
            {row: 9, col: 7, type: 'horse', color: 'red'},
            {row: 9, col: 8, type: 'chariot', color: 'red'},
            {row: 9, col: 1, type: 'cannon', color: 'red'},
            {row: 9, col: 7, type: 'cannon', color: 'red'},
            {row: 6, col: 0, type: 'soldier', color: 'red'},
            {row: 6, col: 2, type: 'soldier', color: 'red'},
            {row: 6, col: 4, type: 'soldier', color: 'red'},
            {row: 6, col: 6, type: 'soldier', color: 'red'},
            {row: 6, col: 8, type: 'soldier', color: 'red'}
        ];
        
        // 清除现有棋子
        document.querySelectorAll('.piece').forEach(piece => piece.remove());
        
        // 放置棋子
        initialSetup.forEach(piece => {
            this.createPiece(piece.row, piece.col, piece.type, piece.color);
        });
    }
    
    createPiece(row, col, type, color) {
        const piece = document.createElement('div');
        piece.className = `piece ${color}`;
        piece.dataset.type = type;
        piece.dataset.color = color;
        piece.dataset.row = row;
        piece.dataset.col = col;
        
        // 设置棋子显示文字
        const pieceNames = {
            'general': color === 'red' ? '帅' : '将',
            'advisor': color === 'red' ? '仕' : '士',
            'elephant': color === 'red' ? '相' : '象',
            'horse': color === 'red' ? '马' : '馬',
            'chariot': color === 'red' ? '车' : '車',
            'cannon': color === 'red' ? '炮' : '砲',
            'soldier': color === 'red' ? '兵' : '卒'
        };
        
        piece.textContent = pieceNames[type];
        
        this.board[row][col].appendChild(piece);
    }
    
    handleSquareClick(row, col) {
        const square = this.board[row][col];
        const piece = square.querySelector('.piece');
        
        // 如果游戏已经结束，不处理点击
        if (this.gameStatus !== 'playing') {
            return;
        }
        
        // 如果已经选中了一个棋子
        if (this.selectedPiece) {
            const fromRow = parseInt(this.selectedPiece.dataset.row);
            const fromCol = parseInt(this.selectedPiece.dataset.col);
            
            // 如果点击的是有效移动位置
            if (this.isValidMovePosition(row, col)) {
                this.movePiece(fromRow, fromCol, row, col);
                this.clearSelection();
                return;
            }
            
            // 如果点击的是自己的其他棋子，改为选择该棋子
            if (piece && piece.dataset.color === this.currentPlayer) {
                this.clearSelection();
                this.selectPiece(piece);
                return;
            }
            
            // 其他情况取消选择
            this.clearSelection();
            return;
        }
        
        // 如果没有选中棋子，且点击位置有棋子
        if (piece) {
            // 如果是当前玩家的棋子，则选中它
            if (piece.dataset.color === this.currentPlayer) {
                this.selectPiece(piece);
            }
        }
    }
    
    selectPiece(piece) {
        this.selectedPiece = piece;
        const row = parseInt(piece.dataset.row);
        const col = parseInt(piece.dataset.col);
        
        // 高亮选中的棋子
        piece.parentElement.classList.add('selected');
        
        // 计算并显示有效移动位置
        this.calculateValidMoves(row, col);
        this.showValidMoves();
    }
    
    clearSelection() {
        if (this.selectedPiece) {
            this.selectedPiece.parentElement.classList.remove('selected');
            this.selectedPiece = null;
        }
        
        this.hideValidMoves();
        this.validMoves = [];
    }
    
    calculateValidMoves(row, col) {
        // 简化版移动规则计算，实际应该根据具体棋子类型计算
        this.validMoves = [];
        const piece = this.board[row][col].querySelector('.piece');
        if (!piece) return;
        
        const type = piece.dataset.type;
        const color = piece.dataset.color;
        
        // 根据不同类型计算有效移动位置
        switch (type) {
            case 'general':
                this.calculateGeneralMoves(row, col, color);
                break;
            case 'advisor':
                this.calculateAdvisorMoves(row, col, color);
                break;
            case 'elephant':
                this.calculateElephantMoves(row, col, color);
                break;
            case 'horse':
                this.calculateHorseMoves(row, col, color);
                break;
            case 'chariot':
                this.calculateChariotMoves(row, col, color);
                break;
            case 'cannon':
                this.calculateCannonMoves(row, col, color);
                break;
            case 'soldier':
                this.calculateSoldierMoves(row, col, color);
                break;
        }
    }
    
    calculateGeneralMoves(row, col, color) {
        // 将/帅的移动范围
        const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        const minRow = color === 'red' ? 7 : 0;
        const maxRow = color === 'red' ? 9 : 2;
        const minCol = 3;
        const maxCol = 5;
        
        for (const [dr, dc] of directions) {
            const newRow = row + dr;
            const newCol = col + dc;
            
            if (newRow >= minRow && newRow <= maxRow && newCol >= minCol && newCol <= maxCol) {
                const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                if (!targetPiece || targetPiece.dataset.color !== color) {
                    this.validMoves.push([newRow, newCol]);
                }
            }
        }
    }
    
    calculateAdvisorMoves(row, col, color) {
        // 士/仕的移动范围
        const directions = [[-1, -1], [-1, 1], [1, -1], [1, 1]];
        const minRow = color === 'red' ? 7 : 0;
        const maxRow = color === 'red' ? 9 : 2;
        const minCol = 3;
        const maxCol = 5;
        
        for (const [dr, dc] of directions) {
            const newRow = row + dr;
            const newCol = col + dc;
            
            if (newRow >= minRow && newRow <= maxRow && newCol >= minCol && newCol <= maxCol) {
                const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                if (!targetPiece || targetPiece.dataset.color !== color) {
                    this.validMoves.push([newRow, newCol]);
                }
            }
        }
    }
    
    calculateElephantMoves(row, col, color) {
        // 象/相的移动范围
        const directions = [[-2, -2], [-2, 2], [2, -2], [2, 2]];
        const riverRow = color === 'red' ? 4 : 5;
        
        for (const [dr, dc] of directions) {
            const newRow = row + dr;
            const newCol = col + dc;
            
            // 不能过河
            if (color === 'red' && newRow < riverRow) continue;
            if (color === 'black' && newRow > riverRow) continue;
            
            // 检查边界
            if (newRow >= 0 && newRow <= 9 && newCol >= 0 && newCol <= 8) {
                // 检查象眼是否被塞住
                const eyeRow = row + dr/2;
                const eyeCol = col + dc/2;
                const eyePiece = this.board[eyeRow][eyeCol].querySelector('.piece');
                
                if (!eyePiece) {
                    const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                    if (!targetPiece || targetPiece.dataset.color !== color) {
                        this.validMoves.push([newRow, newCol]);
                    }
                }
            }
        }
    }
    
    calculateHorseMoves(row, col, color) {
        // 马/馬的移动范围
        const directions = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]];
        const legDirections = [[-1, 0], [-1, 0], [0, -1], [0, 1], [0, -1], [0, 1], [1, 0], [1, 0]];
        
        for (let i = 0; i < directions.length; i++) {
            const [dr, dc] = directions[i];
            const [lr, lc] = legDirections[i];
            
            const newRow = row + dr;
            const newCol = col + dc;
            const legRow = row + lr;
            const legCol = col + lc;
            
            // 检查边界
            if (newRow >= 0 && newRow <= 9 && newCol >= 0 && newCol <= 8) {
                // 检查马腿是否被蹩住
                const legPiece = this.board[legRow][legCol].querySelector('.piece');
                
                if (!legPiece) {
                    const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                    if (!targetPiece || targetPiece.dataset.color !== color) {
                        this.validMoves.push([newRow, newCol]);
                    }
                }
            }
        }
    }
    
    calculateChariotMoves(row, col, color) {
        // 车/車的移动范围
        const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        
        for (const [dr, dc] of directions) {
            let newRow = row + dr;
            let newCol = col + dc;
            
            while (newRow >= 0 && newRow <= 9 && newCol >= 0 && newCol <= 8) {
                const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                
                if (!targetPiece) {
                    // 空位可以移动
                    this.validMoves.push([newRow, newCol]);
                } else {
                    // 有棋子
                    if (targetPiece.dataset.color !== color) {
                        // 敌方棋子可以吃掉
                        this.validMoves.push([newRow, newCol]);
                    }
                    // 无论是敌是我，都不能继续前进
                    break;
                }
                
                newRow += dr;
                newCol += dc;
            }
        }
    }
    
    calculateCannonMoves(row, col, color) {
        // 炮/砲的移动范围
        const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        
        for (const [dr, dc] of directions) {
            let newRow = row + dr;
            let newCol = col + dc;
            let jumped = false; // 是否已经跳过一个棋子
            
            while (newRow >= 0 && newRow <= 9 && newCol >= 0 && newCol <= 8) {
                const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                
                if (!targetPiece) {
                    // 空位
                    if (!jumped) {
                        // 还没有跳过棋子，可以移动
                        this.validMoves.push([newRow, newCol]);
                    }
                } else {
                    // 有棋子
                    if (!jumped) {
                        // 还没有跳过棋子，标记为已跳过
                        jumped = true;
                    } else {
                        // 已经跳过棋子，可以吃子
                        if (targetPiece.dataset.color !== color) {
                            this.validMoves.push([newRow, newCol]);
                        }
                        // 不能再继续前进
                        break;
                    }
                }
                
                newRow += dr;
                newCol += dc;
            }
        }
    }
    
    calculateSoldierMoves(row, col, color) {
        // 兵/卒的移动范围
        let directions = [];
        
        if (color === 'red') {
            // 红方
            if (row > 4) {
                // 未过河，只能向前
                directions = [[-1, 0]];
            } else {
                // 已过河，可以向前或横移
                directions = [[-1, 0], [0, -1], [0, 1]];
            }
        } else {
            // 黑方
            if (row < 5) {
                // 未过河，只能向前
                directions = [[1, 0]];
            } else {
                // 已过河，可以向前或横移
                directions = [[1, 0], [0, -1], [0, 1]];
            }
        }
        
        for (const [dr, dc] of directions) {
            const newRow = row + dr;
            const newCol = col + dc;
            
            // 检查边界
            if (newRow >= 0 && newRow <= 9 && newCol >= 0 && newCol <= 8) {
                const targetPiece = this.board[newRow][newCol].querySelector('.piece');
                if (!targetPiece || targetPiece.dataset.color !== color) {
                    this.validMoves.push([newRow, newCol]);
                }
            }
        }
    }
    
    isValidMovePosition(row, col) {
        return this.validMoves.some(move => move[0] === row && move[1] === col);
    }
    
    showValidMoves() {
        this.validMoves.forEach(([row, col]) => {
            this.board[row][col].classList.add('valid-move');
        });
    }
    
    hideValidMoves() {
        document.querySelectorAll('.square.valid-move').forEach(square => {
            square.classList.remove('valid-move');
        });
    }
    
    movePiece(fromRow, fromCol, toRow, toCol) {
        const piece = this.board[fromRow][fromCol].querySelector('.piece');
        if (!piece) return;
        
        // 移除目标位置的棋子（如果有的话）
        const targetPiece = this.board[toRow][toCol].querySelector('.piece');
        if (targetPiece) {
            targetPiece.remove();
            
            // 检查是否吃掉了将/帅
            if (targetPiece.dataset.type === 'general') {
                this.gameStatus = piece.dataset.color === 'red' ? 'red_won' : 'black_won';
                this.showMessage(`${piece.dataset.color === 'red' ? '红方' : '黑方'}获胜！`, 'success');
            }
        }
        
        // 移动棋子
        this.board[toRow][toCol].appendChild(piece);
        piece.dataset.row = toRow;
        piece.dataset.col = toCol;
        
        // 切换玩家
        this.currentPlayer = this.currentPlayer === 'red' ? 'black' : 'red';
        this.updateGameInfo();
    }
    
    updateGameInfo() {
        const playerElement = document.getElementById('current-player');
        const statusElement = document.getElementById('game-status');
        
        playerElement.textContent = this.currentPlayer === 'red' ? '红方' : '黑方';
        playerElement.className = this.currentPlayer;
        
        statusElement.textContent = this.gameStatus === 'playing' ? '进行中' : 
                                   this.gameStatus === 'red_won' ? '红方获胜' : '黑方获胜';
        statusElement.className = this.gameStatus;
    }
    
    showMessage(text, type) {
        const messageElement = document.getElementById('message');
        messageElement.textContent = text;
        messageElement.className = `message ${type}`;
        
        // 3秒后清除消息
        setTimeout(() => {
            messageElement.textContent = '';
            messageElement.className = 'message';
        }, 3000);
    }
    
    setupEventListeners() {
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetGame();
        });
    }
    
    resetGame() {
        this.currentPlayer = 'red';
        this.gameStatus = 'playing';
        this.selectedPiece = null;
        this.validMoves = [];
        
        this.initializeBoard();
        this.updateGameInfo();
        this.showMessage('游戏已重新开始', 'success');
    }
}

// 页面加载完成后初始化游戏
document.addEventListener('DOMContentLoaded', () => {
    new ChineseChess();
});