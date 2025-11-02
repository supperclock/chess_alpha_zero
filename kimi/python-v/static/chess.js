/* ========== 基础数据 ========== */
const ROWS = 10, COLS = 9;
const PRIMARY_BACKEND_URL = 'https://xq.qzz8io.qzz.io/'; // 主后端服务器URL
const FALLBACK_BACKEND_URL = 'http://127.0.0.1:5000'; // 备用后端服务器URL
let BACKEND_URL = PRIMARY_BACKEND_URL; // 当前使用的后端URL

// board 始终表示"逻辑坐标系"的棋盘：board[logicY][logicX] = piece DOM 或 null
let board = Array.from({length:ROWS},()=>Array(COLS).fill(null));
let currentSide = 'red';      // 红先
let gameOver = false; // 游戏结束标志
let isPaused = false;
let mode = 'human-vs-ai'; // 默认人机对弈
let humanSide = 'red'; // 默认人类执红方
let selectedPiece = null;
let validMoves = [];
let lastMoveFrom = null; // 存储上一步的起点逻辑位置 {x, y}
let movesHistory = []; // 存储走子历史
let recordModePositions = []; // 存储棋谱录制模式下的位置信息
let isBoardFlipped = false; // 视觉是否翻转（只影响显示）
let customMode = 'delete'; // 当前自定义局操作模式
let isFreeMoveMode = false; // 是否处于自由移子模式



/* 初始排布（使用逻辑坐标） */
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

/* ========== 坐标转换（显示 <-> 逻辑） ========== */
// 视觉坐标（display） ↔ 逻辑坐标（logic）互换。
// 我们约定：所有对后端发送的数据与内部逻辑判断采用逻辑坐标。
// 显示位置（DOM left/top）通过 toDisplayCoord(逻辑坐标) 得到。
function toLogicCoord(x, y) {
  if (!isBoardFlipped) return {x, y};
  return { x: COLS - 1 - x, y: ROWS - 1 - y };
}
function toDisplayCoord(x, y) {
  if (!isBoardFlipped) return {x, y};
  return { x: COLS - 1 - x, y: ROWS - 1 - y };
}

/* ====== 后端可用性检查 ====== */
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



/* ====== 初始化与重置 ====== */
function getMode() {
    const select = document.getElementById('mode-select');
    return select ? select.value : 'human-vs-ai';
}

function getHumanSide() {
    const select = document.getElementById('side-select');
    return select ? select.value : 'red';
}

function setModeListener() {
    const select = document.getElementById('mode-select');
    const sideSelection = document.getElementById('side-selection');
    const sideSelect = document.getElementById('side-select');
    const pausecontainer = document.getElementById('pause-container');
    const movebtn = document.getElementById('move-piece-btn');
    
    if (select) {
        select.addEventListener('change', function() {
            mode = select.value;
            // 只有人机对弈模式才显示选择执子方的选项
            if (mode === 'human-vs-ai') {
                sideSelection.style.display = 'inline-block';
                if (pausecontainer) pausecontainer.style.display = 'none'; // 人机对弈不显示暂停按钮
                document.getElementById('custom-define-container').style.display = 'none';
                resetGame();
            } else if (mode === 'ai-vs-ai') {
                sideSelection.style.display = 'none';
                console.log('AI对弈模式已启动');                
                if (pausecontainer) 
                {
                    console.log("显示暂停按钮")
                    pausecontainer.style.display = 'inline-block'; // AI对弈显示暂停按钮"
                }
                resetGame();
            } else if (mode === 'self-define-mode') {
                sideSelection.style.display = 'none';
                if (pausecontainer) pausecontainer.style.display = 'none';
                if (movebtn) movebtn.style.display = 'none';
                document.getElementById('custom-define-container').style.display = 'inline-block';
                resetGameForCustom();
            } else {
                sideSelection.style.display = 'none';
                if (pausecontainer) pausecontainer.style.display = 'none'; // 其他模式不显示暂停按钮
                resetGame();
            }
        });
    }
    
    if (sideSelect) {
        sideSelect.addEventListener('change', function() {
            humanSide = sideSelect.value;
            resetGame();
        });
    }
}

function resetGame() {
    const box = document.getElementById('chessboard');
    // 保存坐标元素（保留棋盘格线与坐标显示）
    const coordinates = box.querySelectorAll('.coordinate');
    const coordinateHTML = Array.from(coordinates).map(coord => coord.outerHTML).join('');
    
    // 清除棋子DOM
    box.innerHTML = '<div class="grid-lines">'+(box.querySelector('.grid-lines') ? box.querySelector('.grid-lines').innerHTML : '')+'</div>' + coordinateHTML;

    board = Array.from({length:ROWS},()=>Array(COLS).fill(null));
    currentSide = 'red';
    gameOver = false;
    isBoardFlipped = false;
    selectedPiece = null;
    validMoves = [];
    lastMoveFrom = null;
    movesHistory = [];
    hideStatus();
    
    // 如果是棋谱录制模式，则重新走 FEN 初始化流程
    if (mode === 'self-define-mode') {
        resetGameForCustom();
        return;
    }
    
    initPieces(); // 默认初始化（使用 initialPieces）
    
    // 根据人类执子方决定是否翻转棋盘
    if (humanSide === 'black' && !isBoardFlipped) {
        flipBoard();
    } else if (humanSide === 'red' && isBoardFlipped) {
        flipBoard();
    }
    
    if (mode === 'ai-vs-ai') {
        setTimeout(aiMove, 500);
    } else if (mode === 'human-vs-ai') {
        // 如果AI先手（人类执黑），让AI先走
        if (humanSide === 'black' && currentSide === 'red') {
            setTimeout(aiMove, 500);
        } else {
            // 启用人类交互的监听
            enableHumanMove();
        }
    } else {
        // 启用人类交互的监听
        enableHumanMove();
    }
    const movePieceBtn = document.getElementById('move-piece-btn');
    const retryBtn = document.getElementById('retry-ai-btn');
    if (movePieceBtn) movePieceBtn.style.display = (mode === 'human-vs-ai') ? 'inline-block' : 'none';
    if (retryBtn) retryBtn.style.display = 'none';
}

/* ====== 初始化棋子（逻辑 -> 显示） ====== */
function initPieces() {
    const box = document.getElementById('chessboard');
    // 清除残留DOM（已在 resetGame 中执行）
    // 放置棋子 DOM，并写入 board（逻辑坐标）
    ['red','black'].forEach(color=>{
      initialPieces[color].forEach(p=>{
        const el = document.createElement('div');
        el.className = `piece ${color}-piece`;
        el.textContent = p.n;
        // p.x, p.y 是逻辑坐标，显示位置按 toDisplayCoord 转换
        const disp = toDisplayCoord(p.x, p.y);
        el.style.left = (25 + disp.x * 50 - 20) + 'px';
        el.style.top  = (25 + disp.y * 50 - 20) + 'px';
        box.appendChild(el);
        board[p.y][p.x] = el; // 存储在逻辑坐标系
      });
    });
    // 移除 enableHumanMove()，让 resetGame 统一控制
}

/* ====== 翻转棋盘（只影响显示） ====== */
function flipBoard() {
    isBoardFlipped = !isBoardFlipped;
    // 重新设置所有 piece 的显示位置（根据其逻辑坐标）
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const piece = board[y][x];
            if (piece) {
                const disp = toDisplayCoord(x, y);
                piece.style.left = (25 + disp.x * 50 - 20) + 'px';
                piece.style.top = (25 + disp.y * 50 - 20) + 'px';
            }
        }
    }
    // 更新上一步提示点的位置（如果存在）
    const moveOrigin = document.querySelector('.move-origin');
    if (moveOrigin && lastMoveFrom) {
        const disp = toDisplayCoord(lastMoveFrom.x, lastMoveFrom.y);
        moveOrigin.style.left = (25 + disp.x * 50 - 6) + 'px';
        moveOrigin.style.top = (25 + disp.y * 50 - 6) + 'px';
    }
    // 清除并重新显示 valid-move-hint（避免坐标混乱）
    clearHints();
    // 如果之前有选中棋子，重新显示它的可走位
    if (selectedPiece) {
        const fromLogic = findPiecePosition(selectedPiece);
        if (fromLogic) {
            validMoves = getValidMoves(fromLogic.x, fromLogic.y);
            showHints(validMoves);
        }
    }
}

/* 自定义局重置 */
function resetGameForCustom() {
  const box = document.getElementById('chessboard');
  box.querySelectorAll('.piece').forEach(p => p.remove());
  board = Array.from({ length: ROWS }, () => Array(COLS).fill(null));
  initPieces();
  enableCustomMode();
}

// 放在全局函数部分（例如靠近其它辅助函数）
async function doHint(side) {
  currentSide = side;
  // side: 'red' 或 'black'
  if (gameOver) {
    showStatus('游戏已结束');
    return;
  }
  showStatus(`${side === 'red' ? '红方' : '黑方'} 正在计算提示...`);
  const boardState = cloneBoardToState(board);
  try {
    let response;
    try {
      response = await fetch(`${BACKEND_URL}/ai_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: boardState, side })
      });
    } catch (err) {
      await checkAndSwitchBackend();
      response = await fetch(`${BACKEND_URL}/ai_move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: boardState, side })
      });
    }
    const move = await response.json();
    if (!move) {
      showStatus('AI 返回空走法（可能认输）');
      return;
    }
    if (move.from && move.to) {
      await tryMove(move);
      showStatus(`${side === 'red' ? '红方' : '黑方'} 完成走棋`);
    } else {
      showStatus('未获取到合法走法');
    }
  } catch (err) {
    console.error('提示失败:', err);
    showStatus('提示失败，查看控制台日志');
  } finally {
    // 保持自定义模式仍然可重复点击 —— 不隐藏按钮，不做自动触发
  }
  
}


/* 自定义局操作逻辑 */
function enableCustomMode() {
  const customSelect = document.getElementById('custom-mode-select');
  const hintBtn = document.getElementById('custom-hint-btn');
  customMode = customSelect ? customSelect.value : 'delete';
  if (customSelect) {
    customSelect.onchange = () => {
      customMode = customSelect.value;
      enableCustomMode();
    };
  }

  document.querySelectorAll('.piece').forEach(el => el.onclick = null);
  const boardBox = document.getElementById('chessboard');
  boardBox.onclick = null;

  // 隐藏提示按钮默认状态
  if (hintBtn) hintBtn.style.display = 'none';
  if (hintBtn) hintBtn.onclick = null;

  if (customMode === 'delete') {
    document.querySelectorAll('.piece').forEach(el => {
      el.onclick = () => {
        const pos = findPiecePosition(el);
        if (pos) {
          board[pos.y][pos.x] = null;
          el.remove();
        }
      };
    });
  } else if (customMode === 'move') {
    enableFreeMove(true);
    // let selected = null;
    // document.querySelectorAll('.piece').forEach(el => {
    //   el.onclick = e => {
    //     e.stopPropagation();
    //     if (selected) selected.classList.remove('move-effect');
    //     selected = el;
    //     selected.classList.add('move-effect');
    //   };
    // });

    // boardBox.onclick = e => {
    //   const rect = boardBox.getBoundingClientRect();
    //   // **【核心修改】**：首先计算出显示坐标系下的位置
    //   const dispX = Math.floor((e.clientX - rect.left - 25) / 50);
    //   const dispY = Math.floor((e.clientY - rect.top - 25) / 50);
      
    //   // 然后将显示坐标转换为逻辑坐标
    //   const logicPos = toLogicCoord(dispX, dispY);
    //   const x = logicPos.x; // 目标逻辑X
    //   const y = logicPos.y; // 目标逻辑Y
    //   if (selected && x >= 0 && x < COLS && y >= 0 && y < ROWS) {
    //     const oldPos = findPiecePosition(selected);
    //     if (oldPos) board[oldPos.y][oldPos.x] = null;
    //     const target = board[y][x];
    //     if (target) return;  // 目标位置有棋子，取消移动
    //     board[y][x] = selected;
    //     const disp = toDisplayCoord(x, y);
    //     selected.style.left = (25 + disp.x * 50 - 20) + 'px';
    //     selected.style.top = (25 + disp.y * 50 - 20) + 'px';
    //     selected.classList.remove('move-effect');
    //     selected = null;
    //   }
    // };
  } else if (customMode === 'hint-red' || customMode === 'hint-black') {
      // 显示提示按钮，让用户每次点击按钮才触发一次提示
      if (hintBtn) {
        hintBtn.style.display = 'inline-block';
        hintBtn.textContent = '提示';

        // 绑定点击事件
        const side = customMode === 'hint-red' ? 'red' : 'black';
        hintBtn.onclick = () => {
          doHint(side);
        };
    }
  }else if (customMode === 'select-black'){
     if (!isBoardFlipped) {
        flipBoard();
     }
  }else if (customMode === 'select-red'){
     if (isBoardFlipped) {
        flipBoard();
     }
  }
}

/* ========== DOM & 坐标辅助函数 ========== */
// 把 piece DOM 在 board 中找到其逻辑坐标（扫描 board）
function findPiecePosition(el) {
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      if (board[y][x] === el) return {x, y};
    }
  }
  return null;
}

// 保留旧的 xy(el) 接口（兼容），但改为返回逻辑坐标
function xy(el){
  const pos = findPiecePosition(el);
  if (pos) return pos;
  // 如果没在 board 中，尝试从样式计算（fallback，按显示坐标计算然后转换为逻辑）
  const b = document.getElementById('chessboard').getBoundingClientRect();
  let dispX = Math.round((el.offsetLeft + 20 - 25) / 50);
  let dispY = Math.round((el.offsetTop  + 20 - 25) / 50);
  return toLogicCoord(dispX, dispY);
}

/* ====== 克隆 board 状态（逻辑坐标）供后端使用 ====== */
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

/* ====== 尝试移动（move 使用逻辑坐标） ====== */
async function tryMove(move) {
    if (gameOver) return;
    if (isPaused) {
        showStatus('已暂停，点击"继续"可恢复');
        return;
    }

    // 期望 move.from / move.to 都是逻辑坐标 {x,y}
    const fromX = move.from.x;
    const fromY = move.from.y;
    const toX = move.to.x;
    const toY = move.to.y;

    // 防护
    if (fromX < 0 || fromX >= COLS || fromY < 0 || fromY >= ROWS) return;
    if (toX < 0 || toX >= COLS || toY < 0 || toY >= ROWS) return;

    // 棋谱录制：保存当前逻辑棋盘和走法
    // if (mode === 'self-define-mode') {
    //     const boardStateBeforeMove = cloneBoardToState(board);
    //     movesHistory.push({
    //         board: boardStateBeforeMove,
    //         side: currentSide,
    //         move: { from: {x: fromX, y: fromY}, to: {x: toX, y: toY} }
    //     });

    //     // 保存到后端数据库（逻辑坐标）
    //     try {
    //         await fetch(`${BACKEND_URL}/save_position`, {
    //             method: 'POST',
    //             headers: { 'Content-Type': 'application/json' },
    //             body: JSON.stringify({ 
    //                 board: boardStateBeforeMove, 
    //                 side: currentSide, 
    //                 move: { from: {x: fromX, y: fromY}, to: {x: toX, y: toY} } 
    //             })
    //         });
    //     } catch (error) {
    //         console.error('保存位置信息失败:', error);
    //     }
    // }

    const pieceEl = board[fromY][fromX];
    if (!pieceEl) return;

    // 检查目标是否己方棋子（不能吃自己）
    const target = board[toY][toX];
    if (target && ((currentSide === 'red' && target.classList.contains('red-piece')) || (currentSide === 'black' && target.classList.contains('black-piece')))) {
        return; // 非法吃子
    }

    // 触发吃子动画（先显示再移除）
    if (target) {
        target.classList.add('boom-effect');
        setTimeout(() => {
            if (target.parentNode) target.parentNode.removeChild(target);
        }, 400);
    }

    // 移动逻辑：更新 board（逻辑）并移动 DOM（显示位置）
    board[fromY][fromX] = null;
    board[toY][toX] = pieceEl;

    const dispTo = toDisplayCoord(toX, toY);
    pieceEl.style.left = (25 + dispTo.x * 50 - 20) + 'px';
    pieceEl.style.top  = (25 + dispTo.y * 50 - 20) + 'px';

    pieceEl.classList.add('move-effect', 'flame-effect');
    setTimeout(() => {
        pieceEl.classList.remove('move-effect', 'flame-effect');
    }, 400);

    // 移除之前的起点提示
    const lastFromDot = document.querySelector('.move-origin');
    if (lastFromDot) {
        lastFromDot.parentNode.removeChild(lastFromDot);
    }

    // 添加新的起点提示（显示位置基于逻辑坐标）
    const fromDot = document.createElement('div');
    fromDot.className = 'move-origin';
    const dispFrom = toDisplayCoord(fromX, fromY);
    fromDot.style.left = (25 + dispFrom.x * 50 - 6) + 'px';
    fromDot.style.top = (25 + dispFrom.y * 50 - 6) + 'px';
    document.getElementById('chessboard').appendChild(fromDot);
    lastMoveFrom = {x: fromX, y: fromY}; // 保存逻辑坐标

    document.querySelectorAll('.piece').forEach(p => p.classList.remove('last-move'));
    pieceEl.classList.add('last-move');

    // 切换行棋方
    currentSide = (currentSide === 'red') ? 'black' : 'red';
    
    // 检查游戏是否结束：发送逻辑棋盘状态给后端
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
        } else if (mode === 'human-vs-ai' && currentSide !== humanSide) {
            // 当前轮到AI行棋时自动调用AI移动
            setTimeout(aiMove, 1000);
        } else if (mode === 'self-define-mode') {
            // 在棋谱录制模式下，启用双方手动操作
            enableHumanMove();  // 这会同时启用红方和黑方的操作
        } else {
            // 常规人机或人对人模式：重新启用点击绑定以便下一步交互
            enableHumanMove();
        }
    }
    clearHints();
}

function enableFreeMove(enable) {
    document.querySelectorAll('.piece').forEach(el => el.onclick = null);
    const boardBox = document.getElementById('chessboard');

    if (!enable) {
        enableHumanMove();
        return;
    }

    let selected = null;
    boardBox.onclick = null;

    document.querySelectorAll('.piece').forEach(el => {
        el.onclick = e => {
            e.stopPropagation();
            if (selected) selected.classList.remove('move-effect');
            selected = el;
            selected.classList.add('move-effect');
        };
    });

    boardBox.onclick = e => {
        if (!selected) return;
        const rect = boardBox.getBoundingClientRect();
        const dispX = Math.floor((e.clientX - rect.left - 25) / 50);
        const dispY = Math.floor((e.clientY - rect.top - 25) / 50);
        const logicPos = toLogicCoord(dispX, dispY);
        const x = logicPos.x, y = logicPos.y;
        if (x < 0 || x >= COLS || y < 0 || y >= ROWS) return;

        const oldPos = findPiecePosition(selected);
        if (oldPos) board[oldPos.y][oldPos.x] = null;
        const target = board[y][x];
        if (target) return;
        board[y][x] = selected;

        const disp = toDisplayCoord(x, y);
        selected.style.left = (25 + disp.x * 50 - 20) + 'px';
        selected.style.top = (25 + disp.y * 50 - 20) + 'px';
        selected.classList.remove('move-effect');
        selected = null;
    };
}


/* ====== AI 移动 ====== */
async function aiMove() {

  if (gameOver) return;
  if (isPaused) {
    showStatus('已暂停，点击"继续"可恢复');
    return;
  }
  showStatus(`${currentSide === 'red' ? '红方' : '黑方'} 正在思考...`);

  const boardState = cloneBoardToState(board); // 逻辑棋盘
  try {
    let response;
    try {
        response = await fetch(`${BACKEND_URL}/ai_move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: boardState, side: currentSide })
        });
    } catch (error) {        
        console.error('AI move error:', error);
        showStatus('AI走棋失败，请点击“重试AI走棋”按钮重试');
        const retryBtn = document.getElementById('retry-ai-btn');
        if (retryBtn) retryBtn.style.display = 'inline-block';       
    }
    const move = await response.json();
    // console.log('AI 移动（逻辑坐标）:', move);
    // 如果 move 为 null，表示 AI 认输
    if (move === null) {
        gameOver = true;
        showStatus(`${currentSide === 'red' ? '红方' : '黑方'} 认输，游戏结束！`);
        return;
    }

    if (move.from && move.to) {
        hideStatus();
        // move.from/to 已经是逻辑坐标，直接传入 tryMove
        tryMove(move);
    }
  } catch (error) {
    console.error('AI move error:', error);
  }
}

/* ====== 状态显示 ====== */
function showStatus(message) {
  const statusDiv = document.getElementById('game-status');
  statusDiv.textContent = message;
  statusDiv.style.display = 'block';
}

function hideStatus() {
  const statusDiv = document.getElementById('game-status');
  statusDiv.style.display = 'none';
}

/* ====== 显示可走提示 & 清理 ====== */
// moves 是逻辑坐标数组 [{x,y}, ...]
function showHints(moves) {
    const box = document.getElementById('chessboard');
    moves.forEach(m => {
        const hint = document.createElement('div');
        hint.className = 'valid-move-hint';
        const disp = toDisplayCoord(m.x, m.y);
        hint.style.left = (25 + disp.x * 50) + 'px';
        hint.style.top = (25 + disp.y * 50) + 'px';
        hint.onclick = function() {
            clearHints();
            if (selectedPiece) {
                const from = findPiecePosition(selectedPiece); // 逻辑坐标
                if (from) {
                    tryMove({from: {x: from.x, y: from.y}, to: {x: m.x, y: m.y}});
                }
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

/* ====== 走法计算（逻辑坐标） ====== */
// getValidMoves 使用逻辑坐标，不依赖显示翻转
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

// canMoveOn 使用逻辑坐标进行所有判断（不再和 isBoardFlipped 纠结）
function canMoveOn(fx, fy, tx, ty, side) {
    const piece = board[fy][fx];
    if (!piece) return false;
    if ((side === 'red' && !piece.classList.contains('red-piece')) || (side === 'black' && !piece.classList.contains('black-piece'))) return false;
    const target = board[ty][tx];
    // 不能吃己方
    if (target && ((side === 'red' && target.classList.contains('red-piece')) || (side === 'black' && target.classList.contains('black-piece')))) return false;
    const name = piece.textContent.trim();
    // 不能原地不动
    if (fx === tx && fy === ty) return false;
    // 不能越界
    if (tx < 0 || tx >= COLS || ty < 0 || ty >= ROWS) return false;
    
    // 走法规则（逻辑坐标）
    if (name === '兵' || name === '卒') {
        // 兵/卒：红往下（y+1），黑往上（y-1）
        let forward = side === 'red' ? 1 : -1;
        let isAcrossRiver = (side === 'red' && fy >= 5) || (side === 'black' && fy <= 4);
        if (tx === fx && ty === fy + forward) return true;
        if (isAcrossRiver && Math.abs(tx - fx) === 1 && ty === fy) return true;
        return false;
    } else if (name === '車' || name === '车') {
        // 车：直线且无阻挡
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
    } else if (name === '馬' || name === '马') {
        // 马：走日，需腿不堵
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
    } else if (name === '炮') {
        // 炮：直线，若不吃子则中间无子；若吃子则中间恰有一个子
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
    } else if (name === '帥' || name === '將') {
        // 帅/将：一格且在九宫内
        let dx = Math.abs(tx - fx), dy = Math.abs(ty - fy);
        if (dx + dy !== 1) return false;
        if (tx < 3 || tx > 5) return false;
        if (side === 'red' && (ty < 0 || ty > 2)) return false;
        if (side === 'black' && (ty < 7 || ty > 9)) return false;
        return true;
    } else if (name === '士' || name === '仕') {
        // 士/仕：斜走一格且在九宫内
        let dx = Math.abs(tx - fx), dy = Math.abs(ty - fy);
        if (dx !== 1 || dy !== 1) return false;
        if (tx < 3 || tx > 5) return false;
        if (side === 'red' && (ty < 0 || ty > 2)) return false;
        if (side === 'black' && (ty < 7 || ty > 9)) return false;
        return true;
    } else if (name === '相' || name === '象') {
        // 相/象：田字走法（2,2），不过河限制，且不能被蹩（中点不能有子）
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

/* ====== 启用人类操作（点击事件绑定） ====== */
function enableHumanMove() {
    // 先解绑所有 piece 的 onclick（防止重复绑定）
    document.querySelectorAll('.piece').forEach(el => el.onclick = null);

    if (mode === 'self-define-mode') {
        // 棋谱录制：双方都可以点击
        document.querySelectorAll('.piece').forEach(el => {
            el.onclick = function(e) {
                if (gameOver || isPaused) return;
                clearHints();
                selectedPiece = el;
                const pos = findPiecePosition(el);
                if (!pos) return;
                // 只显示与当前行棋方一致的选中
                if ((currentSide === 'red' && el.classList.contains('red-piece')) || (currentSide === 'black' && el.classList.contains('black-piece'))) {
                    validMoves = getValidMoves(pos.x, pos.y);
                    showHints(validMoves);
                }
            };
        });
    } else if (mode === 'human-vs-ai') {
        // 人机对弈：根据人类执子方绑定相应颜色的棋子
        const humanPieces = document.querySelectorAll(`.piece.${humanSide}-piece`);
        humanPieces.forEach(el => {
            el.onclick = function(e) {
                if (gameOver || isPaused || mode !== 'human-vs-ai' || currentSide !== humanSide) return;
                clearHints();
                selectedPiece = el;
                const pos = findPiecePosition(el);
                if (!pos) return;
                validMoves = getValidMoves(pos.x, pos.y);
                showHints(validMoves);
            };
        });
    } else {
        // 默认人机模式：只有红方可点击（当前实现是人先红）
        document.querySelectorAll('.piece.red-piece').forEach(el => {
            el.onclick = function(e) {
                if (gameOver || isPaused || mode !== 'human-vs-ai' || currentSide !== 'red') return;
                clearHints();
                selectedPiece = el;
                const pos = findPiecePosition(el);
                if (!pos) return;
                validMoves = getValidMoves(pos.x, pos.y);
                showHints(validMoves);
            };
        });
    }

    // 如果是 self-define-mode 并且黑方需要手动走子，也要单独绑定黑子点击（可选）
    if (mode === 'self-define-mode') {
        enableBlackHumanMove();
    }
}

function enableBlackHumanMove() {
    // 解绑再绑定，避免重复
    document.querySelectorAll('.piece.black-piece').forEach(el => el.onclick = null);
    document.querySelectorAll('.piece.black-piece').forEach(el => {
        el.onclick = function(e) {
            if (gameOver || isPaused || mode !== 'self-define-mode' || currentSide !== 'black') return;
            clearHints();
            selectedPiece = el;
            const pos = findPiecePosition(el);
            if (!pos) return;
            validMoves = getValidMoves(pos.x, pos.y);
            showHints(validMoves);
        };
    });
}

/* ====== 页面初始化 ====== */
function initBoard(){
    setModeListener();
    resetGame();
    const pauseBtn = document.getElementById('pause-btn');
    // ====== 移子功能 ======
    const movePieceBtn = document.getElementById('move-piece-btn');    
    if (movePieceBtn) {
        movePieceBtn.addEventListener('click', () => {
            if (mode !== 'human-vs-ai') {
                showStatus('仅在人机对弈模式下可用');
                return;
            }
            isFreeMoveMode = !isFreeMoveMode;
            movePieceBtn.textContent = isFreeMoveMode ? '退出移子' : '移子';
            // retryBtn.style.display = isFreeMoveMode ? 'inline-block' : 'none';
            showStatus(isFreeMoveMode ? '进入自由移子模式，可拖动棋子调整局面' : '退出移子模式');
            enableFreeMove(isFreeMoveMode);
        });
    }

    if (pauseBtn) {
        pauseBtn.addEventListener('click', function() {
            isPaused = !isPaused;
            pauseBtn.textContent = isPaused ? '继续' : '暂停';
            if (!isPaused && mode === 'ai-vs-ai') {
                setTimeout(aiMove, 1000);
            }
        });
    }
    
    // 初始化时根据模式设置执子方选择框和暂停按钮的显示状态
    const modeSelect = document.getElementById('mode-select');
    const sideSelection = document.getElementById('side-selection');
    const pauseContainer = document.getElementById('pause-container'); // 确保获取到容器
    
    if (modeSelect && sideSelection) {
        // 初始模式
        const initialMode = modeSelect.value;
        if (initialMode === 'human-vs-ai') {
            sideSelection.style.display = 'inline-block';
            if (pauseContainer) pauseContainer.style.display = 'none';
        } else if (initialMode === 'ai-vs-ai') {
            sideSelection.style.display = 'none';
            if (pauseContainer) pauseContainer.style.display = 'inline-block';
        } else {
             sideSelection.style.display = 'none';
             if (pauseContainer) pauseContainer.style.display = 'none';
        }
    }

    const retryBtn = document.getElementById('retry-ai-btn');
    if (retryBtn) {
        retryBtn.addEventListener('click', async () => {
        retryBtn.style.display = 'none';       
        showStatus('重新请求AI走棋中...');
        try {
            await aiMove();
            hideStatus();
        } catch (err) {
            console.error('重试AI失败:', err);
            showStatus('重试AI失败，请检查网络或后端');
            retryBtn.style.display = 'inline-block';
        }
    });
    }
}

window.addEventListener('DOMContentLoaded', initBoard);

/* ====== 其它工具函数（可按需扩展） ====== */
// 例如：导出棋谱、加载棋谱等，这里保留占位
function exportMoves() {
  // 返回 movesHistory 等
  return movesHistory;
}