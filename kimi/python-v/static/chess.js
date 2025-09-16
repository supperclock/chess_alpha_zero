/* ========== 基础数据 ========== */
const ROWS = 10, COLS = 9;
let board = Array.from({length:ROWS},()=>Array(COLS).fill(null)); // 二维数组：存棋子 DOM
let currentSide = 'red';      // 红先
let gameOver = false; // 游戏结束标志
let isPaused = false;

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
    const box = document.getElementById('chessboard');
    /* 创建棋子并登记 board 数组 */
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
    const pauseBtn = document.getElementById('pause-btn');
    pauseBtn.addEventListener('click', function() {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? '继续' : '暂停';
    if (!isPaused) {
        setTimeout(aiMove, 1000);
    }
});
    
    // 游戏开始，AI先走
    aiMove();
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
        // 游戏继续，轮到另一方AI走
        setTimeout(aiMove, 1000);
    }
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