/* ========== 基础数据 ========== */
const ROWS = 10, COLS = 9;
let board = Array.from({length:ROWS},()=>Array(COLS).fill(null)); // 二维数组：存棋子 DOM
let currentSide = 'red';      // 红先
let gameOver = false; // 游戏结束标志
let isPaused = false;
let mode = 'human-vs-ai'; // 默认人机对弈
let selectedPiece = null;
let validMoves = [];

function getMode() {
    const select = document.getElementById('mode-select');
    return select ? select.value : 'human-vs-ai';
}

function setModeListener() {
    const select = document.getElementById('mode-select');
    if (select) {
        select.addEventListener('change', function() {
            mode = select.value;
            resetGame();
        });
    }
}

function resetGame() {
    // 清空棋盘和状态，重新初始化
    const box = document.getElementById('chessboard');
    box.innerHTML = '<div class="grid-lines">'+box.querySelector('.grid-lines').innerHTML+'</div>';
    board = Array.from({length:ROWS},()=>Array(COLS).fill(null));
    currentSide = 'red';
    gameOver = false;
    hideStatus();
    initPieces();
    if (mode === 'ai-vs-ai') {
        setTimeout(aiMove, 500);
    }
}

function initPieces() {
    const box = document.getElementById('chessboard');
    ['red','black'].forEach(color=>{
      initialPieces[color].forEach(p=>{
        const el = document.createElement('div');
        el.className = `piece ${color}-piece`;
        el.textContent = p.n;
        el.style.left = (25+p.x*50-20)+'px';
        el.style.top  = (25+p.y*50-20)+'px';
        box.appendChild(el);
        board[p.y][p.x] = el;
      });
    });
    enableHumanMove();
}

/* 初始排布 */
const initialPieces = {
  red: [
    {n:'帥',x:4,y:0},{n:'仕',x:3,y:0},{n:'仕',x:5,y:0},
    {n:'相',x:2,y:0},{n:'相',x:6,y:0},{n:'馬',x:1,y:0},{n:'馬',x:7,y:0},
    {n:'車',x:0,y:0},{n:'車',x:8,y:0},{n:'炮',x:1,y:2},{n:'炮',x:7,y:2},
    {n:'兵',x:0,y:3},{n:'兵',x:2,y:3},{n:'兵',x:4,y:3},{n:'兵',x:6,y:3},{n:'兵',x:8,y:3}
  ],
  black: [
    {n:'將',x:4,y:9},{n:'士',x:3,y:9},{n:'士',x:5,y:9},
    {n:'象',x:2,y:9},{n:'象',x:6,y:9},{n:'馬',x:1,y:9},{n:'馬',x:7,y:9},
    {n:'車',x:0,y:9},{n:'車',x:8,y:9},{n:'炮',x:1,y:7},{n:'炮',x:7,y:7},
    {n:'卒',x:0,y:6},{n:'卒',x:2,y:6},{n:'卒',x:4,y:6},{n:'卒',x:6,y:6},{n:'卒',x:8,y:6}
  ]
};


/* ========== 游戏流程和渲染 ========== */
function initBoard(){
    setModeListener();
    resetGame();
    const pauseBtn = document.getElementById('pause-btn');
    pauseBtn.addEventListener('click', function() {
        isPaused = !isPaused;
        pauseBtn.textContent = isPaused ? '继续' : '暂停';
        if (!isPaused && mode === 'ai-vs-ai') {
            setTimeout(aiMove, 1000);
        }
    });
}
window.addEventListener('DOMContentLoaded', initBoard);

function xy(el){
  const b = document.getElementById('chessboard').getBoundingClientRect();
  const x = Math.round((el.offsetLeft + 20 - 25) / 50);
  const y = Math.round((el.offsetTop  + 20 - 25) / 50);
  return {x,y};
}

async function tryMove(move) {
    if (gameOver) return;
    if (isPaused) {
        showStatus('已暂停，点击“继续”可恢复');
        return;
    }

    const fromX = move.from.x;
    const fromY = move.from.y;
    const toX = move.to.x;
    const toY = move.to.y;

    const pieceEl = board[fromY][fromX];
    if (!pieceEl) return;

    const target = board[toY][toX];
    if (target) {
        target.classList.add('boom-effect');
        setTimeout(() => {
            if (target.parentNode) target.parentNode.removeChild(target);
        }, 400);
    }

    pieceEl.style.left = (25 + toX * 50 - 20) + 'px';
    pieceEl.style.top  = (25 + toY * 50 - 20) + 'px';
    board[fromY][fromX] = null;
    board[toY][toX] = pieceEl;

    pieceEl.classList.add('move-effect', 'flame-effect');
    setTimeout(() => {
        pieceEl.classList.remove('move-effect', 'flame-effect');
    }, 400);

    document.querySelectorAll('.piece').forEach(p => p.classList.remove('last-move'));
    pieceEl.classList.add('last-move');

    currentSide = (currentSide === 'red') ? 'black' : 'red';
    
    // 检查游戏是否结束
    const boardState = cloneBoardToState(board);
    const checkResponse = await fetch('http://localhost:5000/check_game_over', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: boardState })
    });
    const checkResult = await checkResponse.json();
    if (checkResult.game_over) {
        gameOver = true;
        showStatus(checkResult.message);
    } else {
        // 游戏继续
        if (mode === 'ai-vs-ai') {
            setTimeout(aiMove, 1000);
        } else if (mode === 'human-vs-ai' && currentSide === 'black') {
            setTimeout(aiMove, 1000);
        }
    }
    clearHints();
}

async function aiMove() {
  if (gameOver) return;
  if (isPaused) {
    showStatus('已暂停，点击“继续”可恢复');
    return;
  }
  showStatus(`${currentSide === 'red' ? '红方' : '黑方'} 正在思考...`);

  const boardState = cloneBoardToState(board);
  try {
    const response = await fetch('http://localhost:5000/ai_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: boardState, side: currentSide })
    });
    const move = await response.json();
    // 如果move为null，表示AI认输
    if (move === null) {
        gameOver = true;
        showStatus(`${currentSide === 'red' ? '红方' : '黑方'} 认输，游戏结束！`);
        return;
    }

    if (move.from && move.to) {
        hideStatus();
        tryMove(move);
    }
  } catch (error) {
    console.error('AI move error:', error);
  }
}

function showStatus(message) {
  const statusDiv = document.getElementById('game-status');
  statusDiv.textContent = message;
  statusDiv.style.display = 'block';
}

function hideStatus() {
  const statusDiv = document.getElementById('game-status');
  statusDiv.style.display = 'none';
}


function cloneBoardToState(boardDom) {
  const state = [];
  for (let y = 0; y < ROWS; y++) {
    const row = [];
    for (let x = 0; x < COLS; x++) {
      const p = boardDom[y][x];
      if (!p) {
        row.push(null);
      } else {
        row.push({ 
            type: p.textContent.trim(), 
            side: p.classList.contains('red-piece') ? 'red' : 'black' 
        });
      }
    }
    state.push(row);
  }
  return state;
}

function enableHumanMove() {
    document.querySelectorAll('.piece.red-piece').forEach(el => {
        el.onclick = function(e) {
            if (gameOver || isPaused || mode !== 'human-vs-ai' || currentSide !== 'red') return;
            clearHints();
            selectedPiece = el;
            const pos = xy(el);
            validMoves = getValidMoves(pos.x, pos.y);
            showHints(validMoves);
        };
    });
}

function getValidMoves(x, y) {
    const moves = [];
    for (let ty = 0; ty < ROWS; ty++) {
        for (let tx = 0; tx < COLS; tx++) {
            if (canMoveOn(x, y, tx, ty, 'red')) {
                moves.push({x: tx, y: ty});
            }
        }
    }
    return moves;
}

function showHints(moves) {
    const box = document.getElementById('chessboard');
    moves.forEach(m => {
        const hint = document.createElement('div');
        hint.className = 'valid-move-hint';
        hint.style.left = (25 + m.x * 50) + 'px';
        hint.style.top = (25 + m.y * 50) + 'px';
        hint.onclick = function() {
            clearHints();
            if (selectedPiece) {
                const from = xy(selectedPiece);
                tryMove({from: {x: from.x, y: from.y}, to: {x: m.x, y: m.y}});
                selectedPiece = null;
                validMoves = [];
            }
        };
        box.appendChild(hint);
    });
}

function clearHints() {
    document.querySelectorAll('.valid-move-hint').forEach(h => h.remove());
}

function canMoveOn(fx, fy, tx, ty, side) {
    const piece = board[fy][fx];
    if (!piece || (side === 'red' && !piece.classList.contains('red-piece'))) return false;
    const target = board[ty][tx];
    if (target && target.classList.contains('red-piece')) return false;
    const name = piece.textContent.trim();
    // 不能原地不动
    if (fx === tx && fy === ty) return false;
    // 不能越界
    if (tx < 0 || tx >= COLS || ty < 0 || ty >= ROWS) return false;
    // 走法规则
    if (["兵","卒"].includes(name)) {
        // 兵/卒
        let forward = side === 'red' ? 1 : -1;
        let isAcrossRiver = (side === 'red' && fy >= 5) || (side === 'black' && fy <= 4);
        if (tx === fx && ty === fy + forward) return true;
        if (isAcrossRiver && Math.abs(tx - fx) === 1 && ty === fy) return true;
        return false;
    } else if (["車","车"].includes(name)) {
        // 车
        if (fx === tx) {
            let minY = Math.min(fy, ty), maxY = Math.max(fy, ty);
            for (let i = minY + 1; i < maxY; i++) {
                if (board[i][fx]) return false;
            }
            return true;
        } else if (fy === ty) {
            let minX = Math.min(fx, tx), maxX = Math.max(fx, tx);
            for (let i = minX + 1; i < maxX; i++) {
                if (board[fy][i]) return false;
            }
            return true;
        }
        return false;
    } else if (["馬","马"].includes(name)) {
        // 马
        let dx = Math.abs(tx - fx), dy = Math.abs(ty - fy);
        if (!((dx === 1 && dy === 2) || (dx === 2 && dy === 1))) return false;
        if (dx === 1) {
            let blockY = fy + (ty > fy ? 1 : -1);
            if (board[blockY][fx]) return false;
        } else {
            let blockX = fx + (tx > fx ? 1 : -1);
            if (board[fy][blockX]) return false;
        }
        return true;
    } else if (["炮"].includes(name)) {
        // 炮
        let count = 0;
        if (fx === tx) {
            let minY = Math.min(fy, ty), maxY = Math.max(fy, ty);
            for (let i = minY + 1; i < maxY; i++) {
                if (board[i][fx]) count++;
            }
        } else if (fy === ty) {
            let minX = Math.min(fx, tx), maxX = Math.max(fx, tx);
            for (let i = minX + 1; i < maxX; i++) {
                if (board[fy][i]) count++;
            }
        } else {
            return false;
        }
        if (target) {
            return count === 1;
        } else {
            return count === 0;
        }
    } else if (["帥","將"].includes(name)) {
        // 帅/将
        let dx = Math.abs(tx - fx), dy = Math.abs(ty - fy);
        if (dx + dy !== 1) return false;
        if (tx < 3 || tx > 5) return false;
        if (side === 'red' && (ty < 0 || ty > 2)) return false;
        if (side === 'black' && (ty < 7 || ty > 9)) return false;
        return true;
    } else if (["士","仕"].includes(name)) {
        // 士/仕
        let dx = Math.abs(tx - fx), dy = Math.abs(ty - fy);
        if (dx !== 1 || dy !== 1) return false;
        if (tx < 3 || tx > 5) return false;
        if (side === 'red' && (ty < 0 || ty > 2)) return false;
        if (side === 'black' && (ty < 7 || ty > 9)) return false;
        return true;
    } else if (["相","象"].includes(name)) {
        // 相/象
        let dx = Math.abs(tx - fx), dy = Math.abs(ty - fy);
        if (dx !== 2 || dy !== 2) return false;
        if (side === 'red' && ty > 4) return false;
        if (side === 'black' && ty < 5) return false;
        let blockX = (fx + tx) / 2, blockY = (fy + ty) / 2;
        if (board[blockY][blockX]) return false;
        return true;
    }
    return false;
}