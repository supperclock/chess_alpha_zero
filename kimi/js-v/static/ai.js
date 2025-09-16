// 电脑自动走黑棋
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
// 在 aiMoveBlack 中使用 minimax（以 black 为电脑）
// depth: 搜索深度（2 或 3）
function aiMoveBlack() {
  if(gameOver) return; // 游戏结束禁止AI走棋
  const boardState = cloneBoardToState(board);
  const depth = 2; // 可调：2 快，3 更强但慢
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
