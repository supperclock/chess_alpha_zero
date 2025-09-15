document.addEventListener('DOMContentLoaded', () => {
    const board = document.querySelector('.grid-container');
    const piecesContainer = document.querySelector('.pieces-container');

    // 初始化棋盘格子
    for (let row = 0; row < 10; row++) {
        for (let col = 0; col < 9; col++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            cell.dataset.row = row;
            cell.dataset.col = col;
            board.appendChild(cell);
        }
    }

    // 初始化棋子
    fetch('/reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        renderPieces(data.board);
    });

    function renderPieces(board) {
        piecesContainer.innerHTML = '';
        for (let row = 0; row < board.length; row++) {
            for (let col = 0; col < board[row].length; col++) {
                const piece = board[row][col];
                if (piece) {
                    const pieceElement = document.createElement('div');
                    pieceElement.classList.add('piece');
                    pieceElement.textContent = piece;
                    pieceElement.dataset.row = row;
                    pieceElement.dataset.col = col;
                    piecesContainer.appendChild(pieceElement);
                }
            }
        }
    }
});