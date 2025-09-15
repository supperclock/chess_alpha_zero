// 棋子中文名
const PIECE_NAMES = {
    rC: '车', rM: '马', rX: '相', rS: '仕', rJ: '帅', rP: '炮', rZ: '兵',
    bC: '車', bM: '馬', bX: '象', bS: '士', bJ: '将', bP: '炮', bZ: '卒'
};

// 渲染棋盘
function renderBoard(board) {
    const boardDiv = document.getElementById('chessboard');
    boardDiv.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'chess-row';
        for (let j = 0; j < 9; j++) {
            const cell = document.createElement('div');
            cell.className = 'chess-cell';
            cell.dataset.row = i;
            cell.dataset.col = j;
            if (board[i][j]) {
                const piece = document.createElement('div');
                piece.className = 'chess-piece ' + (board[i][j][0] === 'r' ? 'red' : 'black');
                piece.innerText = PIECE_NAMES[board[i][j]];
                cell.appendChild(piece);
            }
            row.appendChild(cell);
        }
        boardDiv.appendChild(row);
    }
    boardDiv.style.width = (9 * 60) + 'px';
}

// 获取棋盘
function fetchBoard() {
    fetch('/api/board').then(r => r.json()).then(data => {
        renderBoard(data.board);
    });
}

window.onload = fetchBoard;
