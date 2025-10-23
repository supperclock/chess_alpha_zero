/* ========== 基础数据 ========== */
const ROWS = 10, COLS = 9;
const PRIMARY_BACKEND_URL = 'http://localhost:5000'; // 主后端服务器URL
const FALLBACK_BACKEND_URL = 'http://localhost:5000'; // 备用后端服务器URL
let BACKEND_URL = PRIMARY_BACKEND_URL; // 当前使用的后端URL
let board = Array.from({length:ROWS},()=>Array(COLS).fill(null)); // 二维数组：存棋子 DOM
let currentSide = 'red';      // 红先
let gameOver = false; // 游戏结束标志
let isPaused = false;
let mode = 'human-vs-ai'; // 默认人机对弈
let selectedPiece = null;
let validMoves = [];
let lastMoveFrom = null; // 存储上一步的起点位置 {x, y}
let movesHistory = []; // 存储走子历史
let recordModePositions = []; // 存储棋谱录制模式下的位置信息
let isBoardFlipped = false; // 棋盘是否翻转

// 检查并切换后端URL的函数
async function checkAndSwitchBackend() {
    if (BACKEND_URL === FALLBACK_BACKEND_URL) {
        return; // 已经在使用备用URL，无需再检查
    }
    
    try {
        const response = await fetch(`${PRIMARY_BACKEND_URL}/check_game_over`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: [] })
        });
        if (!response.ok) {
            throw new Error('Primary backend not available');
        }
    } catch (error) {
        console.log('主后端服务器不可用，切换到备用服务器:', FALLBACK_BACKEND_URL);
        BACKEND_URL = FALLBACK_BACKEND_URL;
    }
}

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
    // 保存坐标元素
    const coordinates = box.querySelectorAll('.coordinate');
    const coordinateHTML = Array.from(coordinates).map(coord => coord.outerHTML).join('');
    
    box.innerHTML = '<div class="grid-lines">'+box.querySelector('.grid-lines').innerHTML+'</div>' + coordinateHTML;
    board = Array.from({length:ROWS},()=>Array(COLS).fill(null));
    currentSide = 'red';
    gameOver = false;
    isBoardFlipped = false;
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

// 翻转棋盘函数
function flipBoard() {
    isBoardFlipped = !isBoardFlipped;
    
    // 创建一个新的临时棋盘数组
    const newBoard = Array.from({length: ROWS}, () => Array(COLS).fill(null));
    
    // 更新所有棋子的位置和内部board数组
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const piece = board[y][x];
            if (piece) {
                // 计算翻转后的位置
                const flippedX = COLS - 1 - x;
                const flippedY = ROWS - 1 - y;
                
                // 更新棋子位置
                piece.style.left = (25 + flippedX * 50 - 20) + 'px';
                piece.style.top = (25 + flippedY * 50 - 20) + 'px';
                
                // 更新内部board数组
                newBoard[flippedY][flippedX] = piece;
            }
        }
    }
    
    // 替换内部board数组
    board = newBoard;
    
    // 更新上一步提示点的位置（如果存在）
    const moveOrigin = document.querySelector('.move-origin');
    if (moveOrigin && lastMoveFrom) {
        const flippedX = COLS - 1 - lastMoveFrom.x;
        const flippedY = ROWS - 1 - lastMoveFrom.y;
        moveOrigin.style.left = (25 + flippedX * 50 - 6) + 'px';
        moveOrigin.style.top = (25 + flippedY * 50 - 6) + 'px';
    }
}

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
    
    // 添加翻转棋盘按钮事件监听器
    const flipBtn = document.getElementById('flip-btn');
    flipBtn.addEventListener('click', flipBoard);
}

window.addEventListener('DOMContentLoaded', initBoard);

function xy(el){
  const b = document.getElementById('chessboard').getBoundingClientRect();
  let x = Math.round((el.offsetLeft + 20 - 25) / 50);
  let y = Math.round((el.offsetTop  + 20 - 25) / 50);
  
  return {x,y};
}

async function tryMove(move) {
    if (gameOver) return;
    if (isPaused) {
        showStatus('已暂停，点击"继续"可恢复');
        return;
    }

    const fromX = move.from.x;
    const fromY = move.from.y;
    const toX = move.to.x;
    const toY = move.to.y;

    // 只在棋谱录制模式下保存位置信息
    if (mode === 'record-mode') {
        // 保存当前棋盘状态和移动信息
        const boardStateBeforeMove = cloneBoardToState(board);
        movesHistory.push({
            board: boardStateBeforeMove,
            side: currentSide,
            move: move
        });

        // 保存位置信息到后台数据库
        try {
            await fetch(`${BACKEND_URL}/save_position`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    board: boardStateBeforeMove, 
                    side: currentSide, 
                    move: move 
                })
            });
        } catch (error) {
            console.error('保存位置信息失败:', error);
        }
    }

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

    // 移除之前的起点提示
    const lastFromDot = document.querySelector('.move-origin');
    if (lastFromDot) {
        lastFromDot.parentNode.removeChild(lastFromDot);
    }

    // 添加新的起点提示
    const fromDot = document.createElement('div');
    fromDot.className = 'move-origin';
    fromDot.style.left = (25 + fromX * 50 - 6) + 'px';
    fromDot.style.top = (25 + fromY * 50 - 6) + 'px';
    document.getElementById('chessboard').appendChild(fromDot);
    lastMoveFrom = {x: fromX, y: fromY}; // 保存上一步位置

    document.querySelectorAll('.piece').forEach(p => p.classList.remove('last-move'));
    pieceEl.classList.add('last-move');

    currentSide = (currentSide === 'red') ? 'black' : 'red';
    
    // 检查游戏是否结束
    const boardState = cloneBoardToState(board);
    let checkResponse;
    try {
        checkResponse = await fetch(`${BACKEND_URL}/check_game_over`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: boardState })
        });
    } catch (error) {
        // 如果主URL失败，尝试切换到备用URL
        await checkAndSwitchBackend();
        checkResponse = await fetch(`${BACKEND_URL}/check_game_over`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: boardState })
        });
    }
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
        } else if (mode === 'record-mode') {
            // 在棋谱录制模式下，启用双方手动操作
            enableHumanMove();  // 这会同时启用红方和黑方的操作
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
    let response;
    try {
        response = await fetch(`${BACKEND_URL}/ai_move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: boardState, side: currentSide })
        });
    } catch (error) {
        // 如果主URL失败，尝试切换到备用URL
        await checkAndSwitchBackend();
        response = await fetch(`${BACKEND_URL}/ai_move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: boardState, side: currentSide })
        });
    }
    const move = await response.json();
    console.log('AI 移动：', move);
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
    // 在棋谱录制模式下，启用双方手动操作
    if (mode === 'record-mode') {
        document.querySelectorAll('.piece').forEach(el => {
            el.onclick = function(e) {
                if (gameOver || isPaused) return;
                clearHints();
                selectedPiece = el;
                const pos = xy(el);
                                
                // 根据当前行棋方启用相应的移动规则
                if (currentSide === 'red' && el.classList.contains('red-piece')) {
                    validMoves = getValidMoves(pos.x, pos.y);
                    showHints(validMoves);
                } else if (currentSide === 'black' && el.classList.contains('black-piece')) {                
                    validMoves = getValidMoves(pos.x, pos.y);                    
                    showHints(validMoves);
                }
            };
        });
    } else {
        // 原有的人机对弈模式逻辑
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
}

// 为黑方添加手动操作功能
function enableBlackHumanMove() {
    document.querySelectorAll('.piece.black-piece').forEach(el => {
        el.onclick = function(e) {
            if (gameOver || isPaused || mode !== 'record-mode' || currentSide !== 'black') return;
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
    // 获取当前棋子的颜色
    const piece = board[y][x];
    if (!piece) return moves;
    
    const isRedPiece = piece.classList.contains('red-piece');
    const side = isRedPiece ? 'red' : 'black';
    
    for (let ty = 0; ty < ROWS; ty++) {
        for (let tx = 0; tx < COLS; tx++) {
            if (canMoveOn(x, y, tx, ty, side)) {
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

// 修复棋盘翻转后的走法规则
function canMoveOn(fx, fy, tx, ty, side) {
    const piece = board[fy][fx];
    if (!piece || (side === 'red' && !piece.classList.contains('red-piece')) || (side === 'black' && !piece.classList.contains('black-piece'))) return false;
    const target = board[ty][tx];
    // 修正吃子判断逻辑，正确识别己方棋子
    if (target && ((side === 'red' && target.classList.contains('red-piece')) || (side === 'black' && target.classList.contains('black-piece')))) return false;
    const name = piece.textContent.trim();
    // 不能原地不动
    if (fx === tx && fy === ty) return false;
    // 不能越界
    if (tx < 0 || tx >= COLS || ty < 0 || ty >= ROWS) return false;
    
    // 考虑棋盘翻转的情况
    let actualFX = isBoardFlipped ? COLS - 1 - fx : fx;
    let actualFY = isBoardFlipped ? ROWS - 1 - fy : fy;
    let actualTX = isBoardFlipped ? COLS - 1 - tx : tx;
    let actualTY = isBoardFlipped ? ROWS - 1 - ty : ty;
    
    // 【修复】: 不再翻转 side。我们使用原始的 side 和 原始的(actual)坐标系进行判断。
    // let actualSide = side; // <- 这一整块逻辑都删掉
    // if (isBoardFlipped) {
    //     actualSide = side === 'red' ? 'black' : 'red';
    // }
    
    // 走法规则
    if (name === '兵' || name === '卒') {
        // 兵/卒
        let forward = side === 'red' ? 1 : -1; // <-- 【修复】使用 side
        let isAcrossRiver = (side === 'red' && actualFY >= 5) || (side === 'black' && actualFY <= 4); // <-- 【修复】使用 side
        if (actualTX === actualFX && actualTY === actualFY + forward) return true;
        if (isAcrossRiver && Math.abs(actualTX - actualFX) === 1 && actualTY === actualFY) return true;
        return false;
    } else if (name === '車' || name === '车') {
        // 车
        if (actualFX === actualTX) {
            let minY = Math.min(actualFY, actualTY), maxY = Math.max(actualFY, actualTY);
            for (let i = minY + 1; i < maxY; i++) {
                // 需要将检查坐标转换回原始坐标系
                const checkY = isBoardFlipped ? ROWS - 1 - i : i;
                const checkX = isBoardFlipped ? COLS - 1 - actualFX : actualFX;
                if (board[checkY][checkX]) return false;
            }
            return true;
        } else if (actualFY === actualTY) {
            let minX = Math.min(actualFX, actualTX), maxX = Math.max(actualFX, actualTX);
            for (let i = minX + 1; i < maxX; i++) {
                // 需要将检查坐标转换回原始坐标系
                const checkY = isBoardFlipped ? ROWS - 1 - actualFY : actualFY;
                const checkX = isBoardFlipped ? COLS - 1 - i : i;
                if (board[checkY][checkX]) return false;
            }
            return true;
        }
        return false;
    } else if (name === '馬' || name === '马') {
        // 马
        let dx = Math.abs(actualTX - actualFX), dy = Math.abs(actualTY - actualFY);
        if (!((dx === 1 && dy === 2) || (dx === 2 && dy === 1))) return false;
        if (dx === 1) {
            let blockY = actualFY + (actualTY > actualFY ? 1 : -1);
            // 需要将检查坐标转换回原始坐标系
            const checkY = isBoardFlipped ? ROWS - 1 - blockY : blockY;
            const checkX = isBoardFlipped ? COLS - 1 - actualFX : actualFX;
            if (board[checkY][checkX]) return false;
        } else {
            let blockX = actualFX + (actualTX > actualFX ? 1 : -1);
            // 需要将检查坐标转换回原始坐标系
            const checkY = isBoardFlipped ? ROWS - 1 - actualFY : actualFY;
            const checkX = isBoardFlipped ? COLS - 1 - blockX : blockX;
            if (board[checkY][checkX]) return false;
        }
        return true;
    } else if (name === '炮') {
        // 炮
        let count = 0;
        if (actualFX === actualTX) {
            let minY = Math.min(actualFY, actualTY), maxY = Math.max(actualFY, actualTY);
            for (let i = minY + 1; i < maxY; i++) {
                // 需要将检查坐标转换回原始坐标系
                const checkY = isBoardFlipped ? ROWS - 1 - i : i;
                const checkX = isBoardFlipped ? COLS - 1 - actualFX : actualFX;
                if (board[checkY][checkX]) count++;
            }
        } else if (actualFY === actualTY) {
            let minX = Math.min(actualFX, actualTX), maxX = Math.max(actualFX, actualTX);
            for (let i = minX + 1; i < maxX; i++) {
                // 需要将检查坐标转换回原始坐标系
                const checkY = isBoardFlipped ? ROWS - 1 - actualFY : actualFY;
                const checkX = isBoardFlipped ? COLS - 1 - i : i;
                if (board[checkY][checkX]) count++;
            }
        } else {
            return false;
        }
        if (target) {
            return count === 1;
        } else {
            return count === 0;
        }
    } else if (name === '帥' || name === '將') {
        // 帅/将
        let dx = Math.abs(actualTX - actualFX), dy = Math.abs(actualTY - actualFY);
        if (dx + dy !== 1) return false;
        if (actualTX < 3 || actualTX > 5) return false;
        if (side === 'red' && (actualTY < 0 || actualTY > 2)) return false; // <-- 【修复】使用 side
        if (side === 'black' && (actualTY < 7 || actualTY > 9)) return false; // <-- 【修复】使用 side
        return true;
    } else if (name === '士' || name === '仕') {
        // 士/仕
        let dx = Math.abs(actualTX - actualFX), dy = Math.abs(actualTY - actualFY);
        if (dx !== 1 || dy !== 1) return false;
        if (actualTX < 3 || actualTX > 5) return false;
        if (side === 'red' && (actualTY < 0 || actualTY > 2)) return false; // <-- 【修复】使用 side
        if (side === 'black' && (actualTY < 7 || actualTY > 9)) return false; // <-- 【修复】使用 side
        return true;
    } else if (name === '相' || name === '象') {
        // 相/象
        let dx = Math.abs(actualTX - actualFX), dy = Math.abs(actualTY - actualFY);
        if (dx !== 2 || dy !== 2) return false;
        if (side === 'red' && actualTY > 4) return false; // <-- 【修复】使用 side
        if (side === 'black' && actualTY < 5) return false; // <-- 【修复】使用 side
        let blockX = (actualFX + actualTX) / 2, blockY = (actualFY + actualTY) / 2;
        // 需要将检查坐标转换回原始坐标系
        const checkY = isBoardFlipped ? ROWS - 1 - blockY : blockY;
        const checkX = isBoardFlipped ? COLS - 1 - blockX : blockX;
        if (board[checkY][checkX]) return false;
        return true;
    }
    return false;
}