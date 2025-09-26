const PIECE_TO_GLYPH = {
  'r': '車', 'h': '馬', 'c': '炮', 'e': '象', 'a': '士', 'g': '將', 's': '卒',
  'R': '車', 'H': '馬', 'C': '炮', 'E': '相', 'A': '仕', 'G': '帥', 'S': '兵',
  '.': ''
};

const state = {
  board: [],
  side_to_move: 'red',
  human_side: 'red',
  legal_moves: [],
  selected: null,
  ai_thinking: false,
  hovered: null,
  lastMove: null,
  animationFrame: null,
  selectionPulse: 0,
  selectionAnimation: null,
  debugMode: false,
};

const boardCanvas = document.getElementById('board');
const ctx = boardCanvas.getContext('2d');
const statusText = document.getElementById('statusText');

function startSelectionAnimation() {
  if (state.selectionAnimation) {
    cancelAnimationFrame(state.selectionAnimation);
  }
  state.selectionPulse = 0;
  state.selectionAnimation = requestAnimationFrame(() => {
    drawBoard();
  });
}

function stopSelectionAnimation() {
  if (state.selectionAnimation) {
    cancelAnimationFrame(state.selectionAnimation);
    state.selectionAnimation = null;
  }
}

function drawBoard() {
  const width = boardCanvas.width;
  const height = boardCanvas.height;
  ctx.clearRect(0, 0, width, height);

  const marginX = 30;
  const marginY = 30;
  const cellW = (width - marginX * 2) / 8;
  const cellH = (height - marginY * 2) / 9;

  // Board background with gradient
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#f4e4bc');
  gradient.addColorStop(0.5, '#e6d3a3');
  gradient.addColorStop(1, '#d4c4a8');
  ctx.fillStyle = gradient;
  ctx.fillRect(marginX - 10, marginY - 10, width - marginX * 2 + 20, height - marginY * 2 + 20);

  // Add subtle wood grain effect
  ctx.strokeStyle = 'rgba(139, 69, 19, 0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 20; i++) {
    const x = marginX - 10 + Math.random() * (width - marginX * 2 + 20);
    const y = marginY - 10 + Math.random() * (height - marginY * 2 + 20);
    const length = 20 + Math.random() * 40;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + length, y);
    ctx.stroke();
  }

  // Outer border with shadow effect
  ctx.strokeStyle = '#2c1810';
  ctx.lineWidth = 4;
  ctx.strokeRect(marginX, marginY, width - marginX * 2, height - marginY * 2);
  
  // Inner border
  ctx.strokeStyle = '#8b4513';
  ctx.lineWidth = 2;
  ctx.strokeRect(marginX + 2, marginY + 2, width - marginX * 2 - 4, height - marginY * 2 - 4);

  // Horizontal lines with better styling
  ctx.strokeStyle = '#654321';
  ctx.lineWidth = 1.5;
  for (let r = 0; r < 9; r++) {
    const y = marginY + r * cellH;
    ctx.beginPath();
    ctx.moveTo(marginX, y);
    ctx.lineTo(width - marginX, y);
    ctx.stroke();
  }

  // Vertical lines with river gap between ranks 4 and 5
  for (let c = 0; c < 8; c++) {
    const x = marginX + c * cellW;
    ctx.beginPath();
    ctx.moveTo(x, marginY);
    ctx.lineTo(x, marginY + 4 * cellH);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, marginY + 5 * cellH);
    ctx.lineTo(x, height - marginY);
    ctx.stroke();
  }

  // Palace diagonals
  function boardToPx(r, c) {
    const x = marginX + c * cellW;
    const y = marginY + r * cellH;
    return [x, y];
  }
  // Top palace (black): rows 0..2, cols 3..5
  ctx.beginPath();
  let [x1, y1] = boardToPx(0, 3);
  let [x2, y2] = boardToPx(2, 5);
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ;[x1, y1] = boardToPx(0, 5);
  ;[x2, y2] = boardToPx(2, 3);
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  // Bottom palace (red): rows 7..9, cols 3..5
  ctx.beginPath();
  ;[x1, y1] = boardToPx(7, 3);
  ;[x2, y2] = boardToPx(9, 5);
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ;[x1, y1] = boardToPx(7, 5);
  ;[x2, y2] = boardToPx(9, 3);
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  // River text with better styling
  const riverY = marginY + 4.5 * cellH;
  ctx.fillStyle = '#8b4513';
  ctx.font = 'bold ' + Math.floor(Math.min(cellW, cellH) * 0.5) + 'px "Microsoft YaHei", Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  
  // Add shadow to text
  ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
  ctx.fillText('楚河    汉界', width / 2 + 1, riverY + 1);
  ctx.fillStyle = '#8b4513';
  ctx.fillText('楚河    汉界', width / 2, riverY);

  // Corner markers at cannon/soldier points
  function drawCornerMarks(r, c, size, gap) {
    const x = marginX + c * cellW;
    const y = marginY + r * cellH;
    const gx = gap;
    const gy = gap;
    const s = size;
    ctx.lineWidth = 1;
    // four corners around the intersection (inside the cell cross)
    // top-left
    if (c > 0 && r > 0) {
      ctx.beginPath();
      ctx.moveTo(x - gx - s, y - gy);
      ctx.lineTo(x - gx, y - gy);
      ctx.lineTo(x - gx, y - gy - s);
      ctx.stroke();
    }
    // top-right
    if (c < 8 && r > 0) {
      ctx.beginPath();
      ctx.moveTo(x + gx + s, y - gy);
      ctx.lineTo(x + gx, y - gy);
      ctx.lineTo(x + gx, y - gy - s);
      ctx.stroke();
    }
    // bottom-left
    if (c > 0 && r < 9) {
      ctx.beginPath();
      ctx.moveTo(x - gx - s, y + gy);
      ctx.lineTo(x - gx, y + gy);
      ctx.lineTo(x - gx, y + gy + s);
      ctx.stroke();
    }
    // bottom-right
    if (c < 8 && r < 9) {
      ctx.beginPath();
      ctx.moveTo(x + gx + s, y + gy);
      ctx.lineTo(x + gx, y + gy);
      ctx.lineTo(x + gx, y + gy + s);
      ctx.stroke();
    }
  }

  const markSize = Math.min(cellW, cellH) * 0.12;
  const markGap = Math.min(cellW, cellH) * 0.08;
  const soldierCols = [0, 2, 4, 6, 8];
  // Black soldiers at r=3, red soldiers at r=6
  for (const c of soldierCols) {
    drawCornerMarks(3, c, markSize, markGap);
    drawCornerMarks(6, c, markSize, markGap);
  }
  // Cannons: black (2,1)(2,7), red (7,1)(7,7)
  drawCornerMarks(2, 1, markSize, markGap);
  drawCornerMarks(2, 7, markSize, markGap);
  drawCornerMarks(7, 1, markSize, markGap);
  drawCornerMarks(7, 7, markSize, markGap);

  // Enhanced 3D pieces with better styling
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 9; c++) {
      const p = state.board[r]?.[c] ?? '.';
      const cx = marginX + c * cellW;
      const cy = marginY + r * cellH;
      if (p !== '.') {
        const radius = Math.min(cellW, cellH) * 0.38;
        const isRed = /[A-Z]/.test(p);
        const isHovered = state.hovered && state.hovered[0] === r && state.hovered[1] === c;
        const isSelected = state.selected && state.selected[0] === r && state.selected[1] === c;
        
        // Enhanced shadow with multiple layers
        const shadowOffset = isHovered ? 4 : 3;
        const shadowBlur = isHovered ? 8 : 6;
        
        // Outer shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
        ctx.beginPath();
        ctx.ellipse(cx + shadowOffset, cy + shadowOffset, radius + 2, radius + 2, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Inner shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
        ctx.beginPath();
        ctx.ellipse(cx + 1, cy + 1, radius, radius, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Piece background with enhanced 3D gradient
        const pieceGradient = ctx.createRadialGradient(
          cx - radius * 0.3, cy - radius * 0.3, 0,
          cx, cy, radius
        );
        if (isHovered) {
          pieceGradient.addColorStop(0, '#ffffff');
          pieceGradient.addColorStop(0.7, '#f8f8f8');
          pieceGradient.addColorStop(1, '#e8e8e8');
        } else {
          pieceGradient.addColorStop(0, '#ffffff');
          pieceGradient.addColorStop(0.6, '#f5f5f5');
          pieceGradient.addColorStop(1, '#e0e0e0');
        }
        
        ctx.fillStyle = pieceGradient;
        ctx.beginPath();
        ctx.ellipse(cx, cy, radius, radius, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Enhanced border with 3D effect
        const borderGradient = ctx.createLinearGradient(cx - radius, cy - radius, cx + radius, cy + radius);
        borderGradient.addColorStop(0, '#d4af37');
        borderGradient.addColorStop(0.5, '#8b4513');
        borderGradient.addColorStop(1, '#654321');
        
        ctx.strokeStyle = borderGradient;
        ctx.lineWidth = isHovered ? 3 : 2.5;
        ctx.beginPath();
        ctx.ellipse(cx, cy, radius, radius, 0, 0, Math.PI * 2);
        ctx.stroke();
        
        // Inner highlight for 3D effect
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(cx, cy, radius * 0.7, radius * 0.7, 0, 0, Math.PI * 2);
        ctx.stroke();
        
        // Enhanced text with better typography
        const fontSize = Math.floor(Math.min(cellW, cellH) * 0.45);
        ctx.font = `bold ${fontSize}px "Microsoft YaHei", "SimSun", Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Text shadow for depth
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillText(PIECE_TO_GLYPH[p] || p, cx + 1, cy + 1);
        
        // Main text with enhanced colors
        const textColor = isRed ? '#dc143c' : '#2c1810';
        ctx.fillStyle = textColor;
        ctx.fillText(PIECE_TO_GLYPH[p] || p, cx, cy);
        
        // Hover glow effect
        if (isHovered) {
          ctx.strokeStyle = 'rgba(0, 170, 255, 0.4)';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.ellipse(cx, cy, radius + 3, radius + 3, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
  }

  // Enhanced selected piece effects with pulsing animation
  if (state.selected) {
    const [sr, sc] = state.selected;
    const cx = marginX + sc * cellW;
    const cy = marginY + sr * cellH;
    const radius = Math.min(cellW, cellH) * 0.44;
    
    // Pulsing effect
    const pulseIntensity = 0.5 + 0.5 * Math.sin(state.selectionPulse * 0.1);
    const pulseRadius = radius + 8 + pulseIntensity * 6;
    const pulseAlpha = 0.2 + pulseIntensity * 0.3;
    
    // Multiple glow layers for enhanced effect
    for (let i = 3; i >= 1; i--) {
      const layerRadius = pulseRadius - (3 - i) * 2;
      const layerAlpha = pulseAlpha * (0.3 + i * 0.2);
      
      ctx.strokeStyle = `rgba(0, 170, 255, ${layerAlpha})`;
      ctx.lineWidth = Math.max(2, Math.floor(Math.min(cellW, cellH) * 0.08));
      ctx.beginPath();
      ctx.ellipse(cx, cy, layerRadius, layerRadius, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    
    // Inner highlight with pulsing
    ctx.strokeStyle = `rgba(0, 170, 255, ${0.8 + pulseIntensity * 0.2})`;
    ctx.lineWidth = Math.max(3, Math.floor(Math.min(cellW, cellH) * 0.08));
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius, radius, 0, 0, Math.PI * 2);
    ctx.stroke();
    
    // Crown effect - small triangles around the piece
    const crownRadius = radius + 12;
    const crownPoints = 8;
    ctx.fillStyle = `rgba(255, 215, 0, ${0.6 + pulseIntensity * 0.4})`;
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < crownPoints; i++) {
      const angle = (i * 2 * Math.PI) / crownPoints;
      const x1 = cx + Math.cos(angle) * crownRadius;
      const y1 = cy + Math.sin(angle) * crownRadius;
      const x2 = cx + Math.cos(angle + 0.2) * (crownRadius + 4);
      const y2 = cy + Math.sin(angle + 0.2) * (crownRadius + 4);
      const x3 = cx + Math.cos(angle - 0.2) * (crownRadius + 4);
      const y3 = cy + Math.sin(angle - 0.2) * (crownRadius + 4);
      
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x3, y3);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    
    // Update pulse animation
    state.selectionPulse += 1;
  }

  // Enhanced legal moves display
  if (state.selected && state.legal_moves) {
    const [sr, sc] = state.selected;
    const dotRadius = Math.min(cellW, cellH) * 0.1;
    
    for (const move of state.legal_moves) {
      const [from, to] = move;
      const [fr, fc] = from;
      const [tr, tc] = to;
      if (fr === sr && fc === sc) {
        const tx = marginX + tc * cellW;
        const ty = marginY + tr * cellH;
        const isHovered = state.hovered && state.hovered[0] === tr && state.hovered[1] === tc;
        
        // Enhanced shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.ellipse(tx + 2, ty + 2, dotRadius + 1, dotRadius + 1, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Main dot with enhanced gradient
        const dotGradient = ctx.createRadialGradient(
          tx - dotRadius/3, ty - dotRadius/3, 0,
          tx, ty, dotRadius
        );
        if (isHovered) {
          dotGradient.addColorStop(0, '#a8e6a8');
          dotGradient.addColorStop(1, '#32cd32');
        } else {
          dotGradient.addColorStop(0, '#90EE90');
          dotGradient.addColorStop(1, '#228B22');
        }
        
        ctx.fillStyle = dotGradient;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = isHovered ? 2.5 : 2;
        ctx.beginPath();
        ctx.ellipse(tx, ty, dotRadius, dotRadius, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Hover glow for legal moves
        if (isHovered) {
          ctx.strokeStyle = 'rgba(0, 255, 0, 0.6)';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.ellipse(tx, ty, dotRadius + 3, dotRadius + 3, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
  }

  // Debug mode: show coordinate grid
  if (state.debugMode) {
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.3)';
    ctx.lineWidth = 1;
    ctx.font = '12px Arial';
    ctx.fillStyle = 'red';
    
    for (let r = 0; r <= 9; r++) {
      for (let c = 0; c <= 8; c++) {
        const x = marginX + c * cellW;
        const y = marginY + r * cellH;
        
        // Draw intersection point
        ctx.beginPath();
        ctx.ellipse(x, y, 3, 3, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw coordinates
        ctx.fillText(`${r},${c}`, x + 5, y - 5);
      }
    }
  }
}

function getCanvasPoint(evt) {
  const rect = boardCanvas.getBoundingClientRect();
  const scaleX = boardCanvas.width / rect.width;
  const scaleY = boardCanvas.height / rect.height;
  const x = (evt.clientX - rect.left) * scaleX;
  const y = (evt.clientY - rect.top) * scaleY;
  return { x, y };
}

function coordFromEvent(evt) {
  const { x, y } = getCanvasPoint(evt);
  const marginX = 30;
  const marginY = 30;
  const width = boardCanvas.width;
  const height = boardCanvas.height;
  const cellW = (width - marginX * 2) / 8; // 9 columns (0-8) => 8 steps
  const cellH = (height - marginY * 2) / 9; // 10 rows (0-9) => 9 steps

  // Check if point is within board bounds
  if (x < marginX || x > width - marginX || y < marginY || y > height - marginY) {
    return null;
  }

  // Calculate grid position
  const gridX = (x - marginX) / cellW;
  const gridY = (y - marginY) / cellH;
  
  // Find nearest intersection by rounding
  const c = Math.round(gridX);
  const r = Math.round(gridY);
  
  // Validate coordinates (0-9 rows, 0-8 columns)
  if (r < 0 || r > 9 || c < 0 || c > 8) return null;

  // Check if point is close enough to intersection
  const cx = marginX + c * cellW;
  const cy = marginY + r * cellH;
  const dx = x - cx;
  const dy = y - cy;
  const dist = Math.hypot(dx, dy);
  const threshold = Math.min(cellW, cellH) * 0.4; // within piece radius
  
  if (dist > threshold) return null;
  return [r, c];
}

async function fetchState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  Object.assign(state, data);
  statusText.textContent = state.ai_thinking ? 'AI 思考中...' : '就绪';
  drawBoard();
}

async function newGame() {
  const humanSide = document.getElementById('humanSide').value;
  const res = await fetch('/api/new_game', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ human_side: humanSide })
  });
  const data = await res.json();
  Object.assign(state, data);
  statusText.textContent = '新开局';
  drawBoard();
  // If AI moves first or AI vs AI
  if (state.human_side === 'none') {
    // continuous AI vs AI until terminal or user changes
    while (!state.terminal) {
      await aiMove();
      if (state.terminal) break;
    }
  } else if (state.side_to_move !== state.human_side) {
    await aiMove();
  }
}

async function humanMove(fromRC, toRC) {
  // Store last move for animation
  state.lastMove = { from: fromRC, to: toRC };
  
  const res = await fetch('/api/move', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ from: fromRC, to: toRC })
  });
  const data = await res.json();
  if (!data.ok) {
    statusText.textContent = '非法走步';
    state.selected = null;
    state.lastMove = null;
    drawBoard();
    return false;
  }
  Object.assign(state, data);
  state.selected = null;
  stopSelectionAnimation();
  
  // Simple move animation
  animateMove(fromRC, toRC, () => {
    drawBoard();
    if (state.terminal) {
      alert(resultLabel(state.result));
    }
  });
  
  return true;
}

function animateMove(fromRC, toRC, callback) {
  const [fr, fc] = fromRC;
  const [tr, tc] = toRC;
  const marginX = 30;
  const marginY = 30;
  const width = boardCanvas.width;
  const height = boardCanvas.height;
  const cellW = (width - marginX * 2) / 8;
  const cellH = (height - marginY * 2) / 9;
  
  const fromX = marginX + fc * cellW;
  const fromY = marginY + fr * cellH;
  const toX = marginX + tc * cellW;
  const toY = marginY + tr * cellH;
  
  let progress = 0;
  const duration = 200; // ms
  
  function animate() {
    progress += 16; // ~60fps
    const t = Math.min(progress / duration, 1);
    
    // Easing function
    const ease = t * (2 - t);
    
    const currentX = fromX + (toX - fromX) * ease;
    const currentY = fromY + (toY - fromY) * ease;
    
    // Redraw board
    drawBoard();
    
    // Draw moving piece
    if (t < 1) {
      const radius = Math.min(cellW, cellH) * 0.38;
      const p = state.board[tr]?.[tc] ?? '.';
      
      if (p !== '.') {
        // Shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.beginPath();
        ctx.ellipse(currentX + 2, currentY + 2, radius, radius, 0, 0, Math.PI * 2);
        ctx.fill();
        
        // Piece
        const pieceGradient = ctx.createRadialGradient(
          currentX - radius * 0.3, currentY - radius * 0.3, 0,
          currentX, currentY, radius
        );
        pieceGradient.addColorStop(0, '#ffffff');
        pieceGradient.addColorStop(0.6, '#f5f5f5');
        pieceGradient.addColorStop(1, '#e0e0e0');
        
        ctx.fillStyle = pieceGradient;
        ctx.strokeStyle = '#8b4513';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.ellipse(currentX, currentY, radius, radius, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Text
        const isRed = /[A-Z]/.test(p);
        const textColor = isRed ? '#dc143c' : '#2c1810';
        const fontSize = Math.floor(Math.min(cellW, cellH) * 0.45);
        ctx.font = `bold ${fontSize}px "Microsoft YaHei", "SimSun", Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillText(PIECE_TO_GLYPH[p] || p, currentX + 1, currentY + 1);
        ctx.fillStyle = textColor;
        ctx.fillText(PIECE_TO_GLYPH[p] || p, currentX, currentY);
      }
      
      state.animationFrame = requestAnimationFrame(animate);
    } else {
      callback();
    }
  }
  
  animate();
}

function resultLabel(z) {
  if (z === 0) return '平局';
  return z > 0 ? '红胜' : '黑胜';
}

async function aiMove() {
  statusText.textContent = 'AI 思考中...';
  const res = await fetch('/api/ai_move', { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    statusText.textContent = 'AI 错误: ' + data.error;
    return;
  }
  Object.assign(state, data);
  drawBoard();
  if (state.terminal) {
    alert(resultLabel(state.result));
  } else {
    statusText.textContent = '请你走子';
  }
}

async function applySettings() {
  const sims = parseInt(document.getElementById('sims').value, 10);
  const device = document.getElementById('device').value;
  const model_path = document.getElementById('modelPath').value || null;
  const res = await fetch('/api/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sims, device, model_path })
  });
  const data = await res.json();
  if (!data.ok) {
    alert('设置失败: ' + data.error);
    return;
  }
  Object.assign(state, data);
  drawBoard();
}

// Mouse hover tracking
boardCanvas.addEventListener('mousemove', (evt) => {
  const rc = coordFromEvent(evt);
  if (rc) {
    const [r, c] = rc;
    if (!state.hovered || state.hovered[0] !== r || state.hovered[1] !== c) {
      state.hovered = [r, c];
      drawBoard();
    }
  } else if (state.hovered) {
    state.hovered = null;
    drawBoard();
  }
});

boardCanvas.addEventListener('mouseleave', () => {
  if (state.hovered) {
    state.hovered = null;
    drawBoard();
  }
});

boardCanvas.addEventListener('click', async (evt) => {
  if (state.ai_thinking) return;
  if (state.side_to_move !== state.human_side) return;
  
  const rc = coordFromEvent(evt);
  if (!rc) {
    // Click outside board or not close to any intersection
    if (state.selected) {
      state.selected = null;
      stopSelectionAnimation();
      drawBoard();
    }
    return;
  }
  
  const [r, c] = rc;
  const p = state.board[r]?.[c] ?? '.';
  
  if (!state.selected) {
    // Select a piece - only if it's a piece and belongs to human side
    if (p === '.') return;
    const isRed = /[A-Z]/.test(p);
    const isHumanPiece = (state.human_side === 'red' && isRed) || (state.human_side === 'black' && !isRed);
    
    if (!isHumanPiece) return;
    
    state.selected = [r, c];
    startSelectionAnimation();
  } else {
    const [r0, c0] = state.selected;
    
    // If clicking on the same piece, deselect
    if (r0 === r && c0 === c) {
      state.selected = null;
      stopSelectionAnimation();
      drawBoard();
      return;
    }
    
    // If clicking on another piece of the same side, select that piece instead
    if (p !== '.') {
      const isRed = /[A-Z]/.test(p);
      const isHumanPiece = (state.human_side === 'red' && isRed) || (state.human_side === 'black' && !isRed);
      
      if (isHumanPiece) {
        state.selected = [r, c];
        startSelectionAnimation();
        return;
      }
    }
    
    // Try to make the move
    const ok = await humanMove([r0, c0], [r, c]);
    if (ok && !state.terminal && state.side_to_move !== state.human_side) {
      await aiMove();
    }
  }
});

document.getElementById('newGameBtn').addEventListener('click', newGame);
// Removed settings UI

// Debug mode toggle
document.addEventListener('keydown', (evt) => {
  if (evt.key === 'd' || evt.key === 'D') {
    state.debugMode = !state.debugMode;
    drawBoard();
    console.log('Debug mode:', state.debugMode);
  }
});

window.addEventListener('load', async () => {
  await fetchState();
});


