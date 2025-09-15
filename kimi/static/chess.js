/* ========== 基础数据 ========== */
const ROWS = 10, COLS = 9;
let board = Array.from({length:ROWS},()=>Array(COLS).fill(null)); // 二维数组：存棋子 DOM
let currentSide = 'red';      // 红先
let selected  = null;         // 当前被选中的棋子 DOM
let gameOver = false; // 新增：游戏结束标志

/* 初始排布（同你原来写的） */
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

/* ========== 事件绑定 ========== */
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
        /* 只允许红方操作 */
        if(color === 'red') {
          el.addEventListener('click', e=>{
            if(gameOver) return; // 游戏结束禁止操作
            e.stopPropagation();
            if(el.classList.contains(currentSide+'-piece')){
              // 点己方：选中
              hideValidMoves();
              document.querySelectorAll('.piece').forEach(q=>q.style.boxShadow='');
              el.style.boxShadow='0 0 15px gold';
              selected = el;
              showValidMoves(el);
            }else if(selected){
              // 点对方或空位：尝试走子
              const {x,y} = xy(el);
              tryMove(selected, x, y);
            }
          });
        }
      });
    });
    /* 点空白处走子（只允许红方） */
    box.addEventListener('click', e=>{
      if(gameOver) return; // 游戏结束禁止操作
      if(!selected || currentSide !== 'red') return;
      const rect = box.getBoundingClientRect();
      const x = Math.round((e.clientX - rect.left - 25)/50);
      const y = Math.round((e.clientY - rect.top  - 25)/50);
      tryMove(selected, x, y);
    });
  }
window.addEventListener('DOMContentLoaded', initBoard);
  
/* ========== 工具函数 ========== */
function xy(el){                 // 从绝对像素反推格点
  const b = document.getElementById('chessboard').getBoundingClientRect();
  const x = Math.round((el.offsetLeft + 20 - 25) / 50);
  const y = Math.round((el.offsetTop  + 20 - 25) / 50);
  return {x,y};
}
function inBoard(x,y){return x>=0&&x<COLS&&y>=0&&y<ROWS;}

/* ========== 走子流程 ========== */
function tryMove(pieceEl, toX, toY){
  if(gameOver) return; // 游戏结束禁止操作
  const from = {...xy(pieceEl), piece:pieceEl};
  const target = board[toY][toX];
  if(!canMove(from, {x:toX,y:toY}, target)) return; // 不合规则

  /* 吃子 */
  if(target) {
    target.classList.add('boom-effect');
    setTimeout(()=>{
      if(target.parentNode) target.parentNode.removeChild(target);
    }, 400);
  }

  /* 移动 */
  pieceEl.style.left = (25 + toX*50 - 20)+'px';
  pieceEl.style.top  = (25 + toY*50 - 20)+'px';
  board[from.y][from.x] = null;
  board[toY][toX] = pieceEl;

  /* 走子特效 */
  pieceEl.classList.add('move-effect', 'flame-effect');
  setTimeout(()=>{
    pieceEl.classList.remove('move-effect', 'flame-effect');
  }, 400);

  /* 走棋痕迹 */
  document.querySelectorAll('.piece').forEach(p=>p.classList.remove('last-move'));
  pieceEl.classList.add('last-move');

  // === 新增：胜负判断 ===
  if (checkGameOver()) {
    gameOver = true; // 游戏结束
    return;
  }

  // === 新增：将军提示 ===
  const opponent = currentSide === 'red' ? 'black' : 'red';
  if (isInCheck(opponent)) {
    const statusDiv = document.getElementById('game-status');
    statusDiv.textContent = (opponent === 'red' ? '红方' : '黑方') + "被将军！";
    statusDiv.style.display = 'block';
    setTimeout(() => {
      statusDiv.style.display = 'none';
    }, 3000);
  }

  /* 清除提示并切换回合 */
  hideValidMoves();
  currentSide = currentSide==='red'?'black':'red';
  selected = null;
  document.querySelectorAll('.piece').forEach(p=>p.style.boxShadow='');

  // === 新增：电脑自动走黑棋 ===
  if(currentSide === 'black' && !gameOver) {
    setTimeout(()=>{
      aiMoveBlack();
    }, 500);
  }
}
// 在给定 boardState 下找将/帅
function findGeneralInBoardState(boardState, side) {
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const p = boardState[y][x];
      if (!p) continue;
      if ((p.type === '帥' || p.type === '將') && p.side === side) return { x, y };
    }
  }
  return null;
}

function _isInCheck_board(boardState, side) {
  const general = findGeneralInBoardState(boardState, side);
  if (!general) return false;
  const opponent = side === 'red' ? 'black' : 'red';

  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const p = boardState[y][x];
      if (!p || p.side !== opponent) continue;
      if (canMoveOn(boardState, {x,y}, {x:general.x, y: general.y})) return true;
    }
  }
  return false;
}

function _isKingFacingKing_board(boardState) {
  let redKing=null, blackKing=null;
  for (let y = 0; y < ROWS; y++){
    for (let x = 0; x < COLS; x++){
      const p = boardState[y][x];
      if (!p) continue;
      if (p.type === '帥') redKing = {x,y};
      if (p.type === '將') blackKing = {x,y};
    }
  }
  if (redKing && blackKing && redKing.x === blackKing.x) {
    const min = Math.min(redKing.y, blackKing.y), max = Math.max(redKing.y, blackKing.y);
    for (let yy = min+1; yy < max; yy++) if (boardState[yy][redKing.x] !== null) return false;
    return true; // 没被阻挡 -> 对脸
  }
  return false;
}

// 返回 boardState 下某方所有合法走法（用于搜索）
function _getAllLegalMovesInternal_board(boardState, side) {
  const moves = [];
  for (let y = 0; y < ROWS; y++) {
    for (let x = 0; x < COLS; x++) {
      const p = boardState[y][x];
      if (!p || p.side !== side) continue;
      for (let ty = 0; ty < ROWS; ty++) {
        for (let tx = 0; tx < COLS; tx++) {
          const target = boardState[ty][tx];
          if (!canMoveOn(boardState, {x,y}, {x:tx,y:ty}, target)) continue;
          // 模拟移动
          const tmp = boardState.map(r => r.slice());
          tmp[ty][tx] = tmp[y][x];
          tmp[y][x] = null;

          if (!_isInCheck_board(tmp, side)) {
            if (!_isKingFacingKing_board(tmp)) {
              moves.push({ from:{x,y}, to:{x:tx,y:ty} });
            }
          }
        }
      }
    }
  }
  return moves;
}

// 电脑自动走黑棋
// 在 aiMoveBlack 中使用 minimax（以 black 为电脑）
// depth: 搜索深度（2 或 3）
function aiMoveBlack() {
  if(gameOver) return; // 游戏结束禁止AI走棋
  const boardState = cloneBoardToState(board);
  const depth = 4; // 可调：2 快，3 更强但慢
  const best = minimaxRoot(boardState, depth, 'black');
  if (!best) return;
  // 把找到的最好走法映射到 DOM：找到对应的 piece DOM（用 board[][] 当前坐标）
  const pieceDom = board[best.from.y][best.from.x];
  if (!pieceDom) return;
  tryMove(pieceDom, best.to.x, best.to.y);
}

// root：返回最佳 move 对象 {from,to,score}
function minimaxRoot(boardState, depth, side) {
  const moves = getAllLegalMoves_board(boardState, side);
  if (moves.length === 0) return null;
  let bestVal = -Infinity;
  let bestMoves = [];
  for (const m of moves) {
    const tmp = boardState.map(r => r.slice());
    tmp[m.to.y][m.to.x] = tmp[m.from.y][m.from.x];
    tmp[m.from.y][m.from.x] = null;
    const val = minimax(tmp, depth-1, -Infinity, Infinity, (side === 'black' ? 'red' : 'black'));
    if (val > bestVal) { bestVal = val; bestMoves = [m]; }
    else if (val === bestVal) bestMoves.push(m);
  }
  // 随机选择同分走法
  return bestMoves[Math.floor(Math.random()*bestMoves.length)];
}

// minimax 返回 boardState 的评估分（以黑为正）
function minimax(boardState, depth, alpha, beta, sideToMove) {
  if (depth === 0) return evaluateBoard(boardState);

  const moves = getAllLegalMoves_board(boardState, sideToMove);
  if (moves.length === 0) {
    // 无子：若自己被将军则是失败，否则是困毙
    if (_isInCheck_board(boardState, sideToMove)) {
      return sideToMove === 'black' ? -1000000 : 1000000;
    } else {
      return 0;
    }
  }

  if (sideToMove === 'black') {
    let value = -Infinity;
    for (const m of moves) {
      const tmp = boardState.map(r => r.slice());
      tmp[m.to.y][m.to.x] = tmp[m.from.y][m.from.x];
      tmp[m.from.y][m.from.x] = null;
      value = Math.max(value, minimax(tmp, depth-1, alpha, beta, 'red'));
      alpha = Math.max(alpha, value);
      if (alpha >= beta) break; // 剪枝
    }
    return value;
  } else {
    let value = Infinity;
    for (const m of moves) {
      const tmp = boardState.map(r => r.slice());
      tmp[m.to.y][m.to.x] = tmp[m.from.y][m.from.x];
      tmp[m.from.y][m.from.x] = null;
      value = Math.min(value, minimax(tmp, depth-1, alpha, beta, 'black'));
      beta = Math.min(beta, value);
      if (alpha >= beta) break;
    }
    return value;
  }
}

/* ========== 落子合法性检查（完整版） ========== */

// 主函数：根据棋子类型调用相应的合法性检查函数
function canMove(from, to, targetPiece) {
    const fx = from.x, fy = from.y;
    const tx = to.x, ty = to.y;
    const name = from.piece.textContent;
    const side = from.piece.classList.contains('red-piece') ? 'red' : 'black';

    /* 0. 不能出界 */
    if (!inBoard(tx, ty)) return false;

    /* 1. 不能吃自己人 */
    if (targetPiece && targetPiece.classList.contains(side + '-piece')) {
        return false;
    }

    // 根据棋子名称调用不同的规则函数
    switch (name) {
        case '車':
        case '车':
            return canMoveChariot(fx, fy, tx, ty);
        case '炮':
        case '砲':
            const isCapture = targetPiece !== null;
            return canMoveCannon(fx, fy, tx, ty, isCapture);
        case '馬':
        case '马':
            return canMoveHorse(fx, fy, tx, ty);
        case '兵':
        case '卒':
            return canMoveSoldier(fx, fy, tx, ty, side);
        case '帥':
        case '將':
            return canMoveGeneral(fx, fy, tx, ty, side);
        case '相':
        case '象':
            return canMoveElephant(fx, fy, tx, ty, side);
        case '仕':
        case '士':
            return canMoveAdvisor(fx, fy, tx, ty, side);
        default:
            return false;
    }
}

// 俥/車 (Chariot): 任意距离直线移动，路径上不能有其他棋子
function canMoveChariot(fromX, fromY, toX, toY) {
    if (fromX === toX) { // 垂直移动
        const min = Math.min(fromY, toY);
        const max = Math.max(fromY, toY);
        for (let i = min + 1; i < max; i++) {
            if (board[i][fromX] !== null) {
                return false;
            }
        }
    } else if (fromY === toY) { // 水平移动
        const min = Math.min(fromX, toX);
        const max = Math.max(fromX, toX);
        for (let i = min + 1; i < max; i++) {
            if (board[fromY][i] !== null) {
                return false;
            }
        }
    } else {
        return false; // 非直线移动
    }
    return true;
}

// 炮/砲 (Cannon): 移动同车，但吃子必须“隔山打牛”
function canMoveCannon(fromX, fromY, toX, toY, isCapture) {
    let obstacleCount = 0;
    if (fromX === toX) { // 垂直移动
        const min = Math.min(fromY, toY);
        const max = Math.max(fromY, toY);
        for (let i = min + 1; i < max; i++) {
            if (board[i][fromX] !== null) {
                obstacleCount++;
            }
        }
    } else if (fromY === toY) { // 水平移动
        const min = Math.min(fromX, toX);
        const max = Math.max(fromX, toX);
        for (let i = min + 1; i < max; i++) {
            if (board[fromY][i] !== null) {
                obstacleCount++;
            }
        }
    } else {
        return false; // 非直线移动
    }

    if (isCapture) {
        return obstacleCount === 1; // 吃子必须隔一个子
    } else {
        return obstacleCount === 0; // 走空位不能隔子
    }
}

// 傌/馬 (Horse): 走“日”字，不能“蹩马腿”
function canMoveHorse(fromX, fromY, toX, toY) {
    const dx = Math.abs(toX - fromX);
    const dy = Math.abs(toY - fromY);
    if ((dx === 1 && dy === 2) || (dx === 2 && dy === 1)) {
        // 检查“蹩马腿”
        if (dx === 1) { // 水平走一步
            const checkY = (toY > fromY) ? fromY + 1 : fromY - 1;
            if (board[checkY][fromX] !== null) {
                return false;
            }
        } else { // 垂直走一步
            const checkX = (toX > fromX) ? fromX + 1 : fromX - 1;
            if (board[fromY][checkX] !== null) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// 兵/卒 (Soldier): 只能前进，过河后可左右移动
function canMoveSoldier(fromX, fromY, toX, toY, side) {
    const dx = Math.abs(toX - fromX);
    const dy = toY - fromY;
    const isAcrossRiver = (side === 'red' && fromY >= 5) || (side === 'black' && fromY <= 4);

    // 卒不能后退
    if ((side === 'red' && dy < 0) || (side === 'black' && dy > 0)) {
        return false;
    }

    // 必须走一步
    if (dx + Math.abs(dy) !== 1) {
        return false;
    }

    if (isAcrossRiver) {
        return dx <= 1 && dy >= 0; // 过河后可左右或前进
    } else {
        return dx === 0 && (dy > 0 && side === 'red' || dy < 0 && side === 'black'); // 没过河只能前进
    }
}

// 將/帥 (General): 只能在九宫格内直行一步，不能“对脸”
function canMoveGeneral(fromX, fromY, toX, toY, side) {
    const dx = Math.abs(toX - fromX);
    const dy = Math.abs(toY - fromY);

    // 只能直行一步
    if (dx + dy !== 1) {
        return false;
    }

    // 必须在九宫格内
    if (toX < 3 || toX > 5) return false;
    if (side === 'red' && toY > 2) return false;
    if (side === 'black' && toY < 7) return false;

    // 检查“对脸”规则
    if (fromX === toX) {
        let opponentGeneralY = -1;
        for (let y = 0; y < ROWS; y++) {
            const piece = board[y][toX];
            if (piece && (piece.textContent === '將' || piece.textContent === '帥') && piece.classList.contains((side === 'red' ? 'black' : 'red') + '-piece')) {
                opponentGeneralY = y;
                break;
            }
        }
        if (opponentGeneralY !== -1) {
            let isBlocked = false;
            const min = Math.min(toY, opponentGeneralY);
            const max = Math.max(toY, opponentGeneralY);
            for (let i = min + 1; i < max; i++) {
                if (board[i][toX] !== null) {
                    isBlocked = true;
                    break;
                }
            }
            if (!isBlocked) {
                return false;
            }
        }
    }

    return true;
}

// 相/象 (Elephant): 走“田”字，不能过河，不能“塞象眼”
function canMoveElephant(fromX, fromY, toX, toY, side) {
    const dx = Math.abs(toX - fromX);
    const dy = Math.abs(toY - fromY);

    if (dx !== 2 || dy !== 2) return false; // 非“田”字移动

    // 不能过河
    const isRedAcrossRiver = side === 'red' && toY > 4;
    const isBlackAcrossRiver = side === 'black' && toY < 5;
    if (isRedAcrossRiver || isBlackAcrossRiver) return false;

    // 不能“塞象眼”
    const midX = (fromX + toX) / 2;
    const midY = (fromY + toY) / 2;
    if (board[midY][midX] !== null) return false;
    
    return true;
}

// 士/仕 (Advisor): 只能在九宫格内斜行一步
function canMoveAdvisor(fromX, fromY, toX, toY, side) {
    const dx = Math.abs(toX - fromX);
    const dy = Math.abs(toY - fromY);
    if (dx !== 1 || dy !== 1) return false; // 非斜行一步
    // 必须在九宫格内
    if (toX < 3 || toX > 5) return false;
    if (side === 'red' && toY > 2) return false;
    if (side === 'black' && toY < 7) return false;
    return true;
}

function showValidMoves(pieceEl) {
    const from = {...xy(pieceEl), piece: pieceEl};
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const target = board[y][x];
            if (canMove(from, {x: x, y: y}, target)) {
                // 如果目标位置是空的，创建一个高亮圆点
                if (!target) {
                    const hint = document.createElement('div');
                    hint.className = 'valid-move-hint';
                    hint.style.left = (25 + x * 50) + 'px';
                    hint.style.top = (25 + y * 50) + 'px';
                    document.getElementById('chessboard').appendChild(hint);
                    // 点击提示点可以走子
                    hint.addEventListener('click', () => tryMove(pieceEl, x, y));
                }
                // 如果目标位置是敌方棋子，改变其边框颜色
                else {
                    target.style.outline = '3px solid #00ff00';
                    target.style.outlineOffset = '-3px';
                }
            }
        }
    }
}

// 隐藏所有可走位置提示
function hideValidMoves() {
    // 移除所有高亮圆点
    document.querySelectorAll('.valid-move-hint').forEach(el => el.remove());
    // 移除所有敌方棋子的边框高亮
    document.querySelectorAll('.piece').forEach(el => el.style.outline = '');
}
// 基于 boardState（纯数据）判断走法是否合规
function canMoveOn(boardState, from, to, targetPieceObj) {
  const fx = from.x, fy = from.y;
  const tx = to.x, ty = to.y;
  const pieceObj = boardState[fy][fx];
  if (!pieceObj) return false;
  const name = pieceObj.type;
  const side = pieceObj.side;

  function inBoardCoords(x,y){ return x>=0 && x < COLS && y >=0 && y < ROWS; }
  if (!inBoardCoords(tx,ty)) return false;

  // 不能吃自己人
  const target = boardState[ty][tx];
  if (target && target.side === side) return false;

  // 逐个分发到具体规则函数（使用 boardState 而不是 DOM）
  switch (name) {
    case '車': case '俥': case '车':
      return canMoveChariotOn(boardState, fx,fy,tx,ty);
    case '炮': case '砲':
      return canMoveCannonOn(boardState, fx,fy,tx,ty, !!target);
    case '馬': case '傌': case '马':
      return canMoveHorseOn(boardState, fx,fy,tx,ty);
    case '兵': case '卒':
      return canMoveSoldierOn(boardState, fx,fy,tx,ty, side);
    case '帥': case '將':
      return canMoveGeneralOn(boardState, fx,fy,tx,ty, side);
    case '相': case '象':
      return canMoveElephantOn(boardState, fx,fy,tx,ty, side);
    case '仕': case '士':
      return canMoveAdvisorOn(boardState, fx,fy,tx,ty, side);
    default:
      return false;
  }
}

// 然后实现每个子类型的 boardState 版本（逻辑与原函数相同，但引用 boardState）：
function canMoveChariotOn(boardState, fromX, fromY, toX, toY) {
  if (fromX === toX) {
    const min = Math.min(fromY, toY), max = Math.max(fromY, toY);
    for (let i = min+1; i < max; i++) if (boardState[i][fromX] !== null) return false;
    return true;
  } else if (fromY === toY) {
    const min = Math.min(fromX, toX), max = Math.max(fromX, toX);
    for (let i = min+1; i < max; i++) if (boardState[fromY][i] !== null) return false;
    return true;
  }
  return false;
}

function canMoveCannonOn(boardState, fromX, fromY, toX, toY, isCapture) {
  let obstacleCount = 0;
  if (fromX === toX) {
    const min = Math.min(fromY, toY), max = Math.max(fromY, toY);
    for (let i = min+1; i < max; i++) if (boardState[i][fromX] !== null) obstacleCount++;
  } else if (fromY === toY) {
    const min = Math.min(fromX, toX), max = Math.max(fromX, toX);
    for (let i = min+1; i < max; i++) if (boardState[fromY][i] !== null) obstacleCount++;
  } else return false;

  return isCapture ? (obstacleCount === 1) : (obstacleCount === 0);
}

function canMoveHorseOn(boardState, fromX, fromY, toX, toY) {
  const dx = Math.abs(toX - fromX), dy = Math.abs(toY - fromY);
  if (!((dx === 1 && dy === 2) || (dx === 2 && dy === 1))) return false;
  // 检查蹩马腿
  if (dx === 1) {
    const checkY = (toY > fromY) ? fromY + 1 : fromY - 1;
    if (boardState[checkY][fromX] !== null) return false;
  } else {
    const checkX = (toX > fromX) ? fromX + 1 : fromX - 1;
    if (boardState[fromY][checkX] !== null) return false;
  }
  return true;
}

function canMoveSoldierOn(boardState, fromX, fromY, toX, toY, side) {
  const dx = Math.abs(toX - fromX);
  const dy = toY - fromY;
  const isAcrossRiver = (side === 'red' && fromY >= 5) || (side === 'black' && fromY <= 4);

  // 不能后退
  if ((side === 'red' && dy < 0) || (side === 'black' && dy > 0)) return false;
  if (dx + Math.abs(dy) !== 1) return false;

  if (isAcrossRiver) {
    // 过河后可左右或前进（dy 对 red 是正）
    if (side === 'red') return (dy === 1 && dx === 0) || (dy === 0 && dx === 1);
    else return (dy === -1 && dx === 0) || (dy === 0 && dx === 1);
  } else {
    // 没过河只允许向前
    if (side === 'red') return dx === 0 && dy === 1;
    else return dx === 0 && dy === -1;
  }
}

function canMoveGeneralOn(boardState, fromX, fromY, toX, toY, side) {
  const dx = Math.abs(toX - fromX), dy = Math.abs(toY - fromY);
  if (dx + dy !== 1) return false;
  if (toX < 3 || toX > 5) return false;
  if (side === 'red' && toY > 2) return false;
  if (side === 'black' && toY < 7) return false;

  // 对脸检查（如果与对方将同列且中间无子则不允许）
  if (fromX === toX) {
    let opponentGeneralY = -1;
    for (let y = 0; y < ROWS; y++) {
      const p = boardState[y][toX];
      if (p && (p.type === '將' || p.type === '帥') && p.side !== side) {
        opponentGeneralY = y; break;
      }
    }
    if (opponentGeneralY !== -1) {
      let blocked = false;
      const min = Math.min(toY, opponentGeneralY), max = Math.max(toY, opponentGeneralY);
      for (let i = min+1; i < max; i++) if (boardState[i][toX] !== null) { blocked = true; break; }
      if (!blocked) return false;
    }
  }
  return true;
}

function canMoveElephantOn(boardState, fromX, fromY, toX, toY, side) {
  const dx = Math.abs(toX - fromX), dy = Math.abs(toY - fromY);
  if (dx !== 2 || dy !== 2) return false;
  const isRedAcrossRiver = side === 'red' && toY > 4;
  const isBlackAcrossRiver = side === 'black' && toY < 5;
  if (isRedAcrossRiver || isBlackAcrossRiver) return false;
  const midX = (fromX + toX) / 2, midY = (fromY + toY) / 2;
  if (boardState[midY][midX] !== null) return false;
  return true;
}

function canMoveAdvisorOn(boardState, fromX, fromY, toX, toY, side) {
  const dx = Math.abs(toX - fromX), dy = Math.abs(toY - fromY);
  if (dx !== 1 || dy !== 1) return false;
  if (toX < 3 || toX > 5) return false;
  if (side === 'red' && toY > 2) return false;
  if (side === 'black' && toY < 7) return false;
  return true;
}


/* ========== 新增：将军检测与胜负判断 ========== */

// 找到一方的将/帅
function findGeneral(side) {
    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const piece = board[y][x];
            if (!piece) continue;
            if (piece.textContent === '將' || piece.textContent === '帥') {
                const pieceSide = piece.classList.contains('red-piece') ? 'red' : 'black';
                if (pieceSide === side) {
                    return { x, y, piece };
                }
            }
        }
    }
    return null;
}

// 判断某方是否被将军
function isInCheck(side) {
    const general = findGeneral(side);
    if (!general) return false; // 已经没了

    const opponent = side === 'red' ? 'black' : 'red';

    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const piece = board[y][x];
            if (!piece) continue;
            if (!piece.classList.contains(opponent + '-piece')) continue;

            if (canMove({ x, y, piece }, { x: general.x, y: general.y }, general.piece)) {
                return true;
            }
        }
    }
    return false;
}

// 检查胜负
function checkGameOver() {
    const redGeneral = findGeneral('red');
    const blackGeneral = findGeneral('black');

    if (!redGeneral) {
        const statusDiv = document.getElementById('game-status');
        statusDiv.textContent = "黑方胜利！";
        statusDiv.style.display = 'block';
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
        return true;
    }
    if (!blackGeneral) {
        const statusDiv = document.getElementById('game-status');
        statusDiv.textContent = "红方胜利！";
        statusDiv.style.display = 'block';
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 3000);
        return true;
    }
    return false;
}

// 将当前 DOM board（board[][] 存 DOM 或 null）转换成轻量 boardState（纯数据）
// boardState[y][x] = null 或 { type: '車', side: 'red' }
function cloneBoardToState(boardDom) {
  const state = [];
  for (let y = 0; y < ROWS; y++) {
    const row = [];
    for (let x = 0; x < COLS; x++) {
      const p = boardDom[y][x];
      if (!p) {
        row.push(null);
      } else {
        // 标准化棋子名称（取 textContent 的第一个字符）
        row.push({ type: p.textContent.trim(), side: p.classList.contains('red-piece') ? 'red' : 'black' });
      }
    }
    state.push(row);
  }
  return state;
}

/* ========== 新增：AI评估函数与辅助函数 ========== */

// 统一 piece value 表（建议使用标准中文名称）
const PIECE_VALUES_STD = {
  '將': 10000, '帥': 10000,
  '車': 900, '俥': 900, '车': 900,
  '馬': 450, '傌':450, '马':450,
  '炮': 400, '砲':400,
  '相': 200, '象':200,
  '仕': 200, '士':200,
  '兵': 100, '卒':100
};

// 根据 boardState 评估局面，返回 对 black - red 的分数（或你想要的方向）
// 我这里返回 black 的分数减 red 的分数（positive 代表黑方有利）
function evaluateBoard(boardState) {
  let score = 0;
  for (let y=0; y<ROWS; y++){
    for (let x=0; x<COLS; x++){
      const p = boardState[y][x];
      if (!p) continue;
      const val = PIECE_VALUES_STD[p.type] || 0;
      const sideMult = (p.side === 'black') ? 1 : -1;
      let add = val * sideMult;

      // 兵过河微加分（鼓励进攻）
      if ((p.type === '兵' || p.type === '卒')) {
        const isAcross = (p.side === 'red' && y >= 5) || (p.side === 'black' && y <= 4);
        if (isAcross) add += 20 * sideMult;
      }

      // 中路优先（简单位置分）
      if (x >= 3 && x <= 5) add += 5 * sideMult;

      score += add;
    }
  }

  // 将军惩罚/奖励（若某方被将军，降低其分数）
  if (_isInCheck_board(boardState, 'red')) score += 200; // 黑方受益
  if (_isInCheck_board(boardState, 'black')) score -= 200;

  return score;
}

// 评估单步走法后得分（用于单步排序/快速评估）
// 返回 move 的评估分（越大越优先给 black）
function scoreMove_board(from, to, boardState) {
  const tmp = boardState.map(r => r.slice());
  const moving = tmp[from.y][from.x];
  const captured = tmp[to.y][to.x];

  // 简单 SEE: 捕获价值
  let base = 0;
  if (captured) base += PIECE_VALUES_STD[captured.type] || 0;

  // 执行移动
  tmp[to.y][to.x] = moving;
  tmp[from.y][from.x] = null;

  // 如果移动后把自己置于被将军状态，给予很低分
  if (_isInCheck_board(tmp, moving.side)) return -999999;

  // 额外：若走后对方被将军，奖励
  const opponent = (moving.side === 'black') ? 'red' : 'black';
  let checkBonus = 0;
  if (_isInCheck_board(tmp, opponent)) checkBonus = 150;

// 额外：若走后对方将/帅被吃掉，极大奖励
  const opponentGeneral = (opponent === 'black') ? '將' : '帥';
  let generalCaptured = false;
  for (let y=0; y<ROWS; y++){
    for (let x=0; x<COLS; x++){
      const p = tmp[y][x];
      if (p && p.type === opponentGeneral && p.side === opponent) {
        generalCaptured = true;
        break;
      }
    }
    if (generalCaptured) break;
  }
  if (generalCaptured) checkBonus += 10000;

  // 返回评估：以黑方视角为正（所以要 evaluateBoard(tmp)）
  return evaluateBoard(tmp) + checkBonus + base;
}

// 生成 boardState 下的所有合法走法（附带 score，用于排序）
function getAllLegalMoves_board(boardState, side) {
  const moves = [];
  for (let y=0; y<ROWS; y++){
    for (let x=0; x<COLS; x++){
      const p = boardState[y][x];
      if (!p || p.side !== side) continue;
      for (let ty=0; ty<ROWS; ty++){
        for (let tx=0; tx<COLS; tx++){
          if (!canMoveOn(boardState, {x,y}, {x:tx,y:ty})) continue;
          const tmp = boardState.map(r => r.slice());
          tmp[ty][tx] = tmp[y][x];
          tmp[y][x] = null;
          if (_isInCheck_board(tmp, side)) continue;
          if (_isKingFacingKing_board(tmp)) continue;
          const sc = scoreMove_board({x,y}, {x:tx,y:ty}, boardState);
          moves.push({ from:{x,y}, to:{x:tx,y:ty}, score: sc });
        }
      }
    }
  }
  // 按 score 降序（对 black 来说大分好）
  moves.sort((a,b)=>b.score - a.score);
  return moves;
}


/* ========== 主函数：生成所有合法走子（修改版）========== */
/**
 * 返回在任意棋面下所有合法的走棋集合，并附带分数
 * @returns {Array<Object>} 一个包含所有合法走法的数组，每个走法包含起点、终点和分数
 */
function getAllLegalMoves() {
    const legalMoves =[];

    for (let y = 0; y < ROWS; y++) {
        for (let x = 0; x < COLS; x++) {
            const piece = board[y][x];
            if (!piece ||!piece.classList.contains(currentSide + '-piece')) {
                continue;
            }

            for (let ty = 0; ty < ROWS; ty++) {
                for (let tx = 0; tx < COLS; tx++) {
                    const targetPiece = board[ty][tx];
                    
                    if (canMove({ x, y, piece }, { x: tx, y: ty }, targetPiece)) {
                        
                        const tempBoard = [...board.map(row => [...row])];
                        const originalPiece = tempBoard[y][x];
                        const originalTarget = tempBoard[ty][tx];

                        tempBoard[ty][tx] = tempBoard[y][x];
                        tempBoard[y][x] = null;

                        const isCheckAfterMove = _isInCheck(tempBoard, currentSide);
                        const isKingFacingAfterMove = _isKingFacingKing(tempBoard);

                        if (!isCheckAfterMove &&!isKingFacingAfterMove) {
                            const score = scoreMove({ x, y, piece }, { x: tx, y: ty }, tempBoard);
                            legalMoves.push({
                                from: { x, y },
                                to: { x: tx, y: ty },
                                score: score
                            });
                        }

                        tempBoard[y][x] = originalPiece;
                        tempBoard[ty][tx] = originalTarget;
                    }
                }
            }
        }
    }

    return legalMoves;
}