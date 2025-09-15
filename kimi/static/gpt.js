// 调用 Ollama API 生成解说
async function generateCommentary(from, to, target) {
  const pieceName = from.piece.textContent;
  const side = from.piece.classList.contains('red-piece') ? '红方' : '黑方';
  const moveDesc = `${side}${pieceName} 从(${from.x},${from.y}) 走到 (${to.x},${to.y})` + (target ? `，吃掉了${target.textContent}` : '');
  const prompt = `请用形象的故事性语言解说这步中国象棋，不要带数字，只讲故事：${moveDesc}/no_think`;
  
  const responseContent = document.getElementById('response-content');
  
  // 添加分隔符和当前走步信息
  const moveInfo = `\n\n--- ${moveDesc} ---\n`;
  responseContent.innerHTML += moveInfo;
  responseContent.innerHTML += '<div class="current-commentary">正在生成解说...</div>';

  const es = new EventSource(`/api/generate?prompt=${encodeURIComponent(prompt)}`);

  let responseText = '';
  // 获取当前刚创建的解说div
  const currentCommentaryDiv = responseContent.querySelector('.current-commentary:last-child');
  
  // 将文本中的转义序列规范化为真实字符（如 \n -> 换行）
  function normalizeEscapes(text) {
    if (!text) return '';
    return text
      .replace(/\\n/g, '\n')
      .replace(/\\t/g, '\t')
      .replace(/\n{2,}/g, '\n'); // 将2个或更多连续换行符缩减为1个
  }

  // 配置 marked 支持单换行自动换行
  if (window.marked && typeof marked.setOptions === 'function') {
    marked.setOptions({ breaks: true, gfm: true });
  }
  
  es.onmessage = (event) => {
    try {
      
      // 处理可能的 bytes 格式数据
      let dataStr = event.data;
      if (typeof dataStr === 'string' && dataStr.startsWith("b'")) {
        // 去掉 b' 前缀和 ' 后缀
        dataStr = dataStr.slice(2, -1);
        // 解码 UTF-8 字节序列 - 先处理3字节的中文字符
        dataStr = dataStr.replace(/\\x([0-9a-fA-F]{2})\\x([0-9a-fA-F]{2})\\x([0-9a-fA-F]{2})/g, (match, b1, b2, b3) => {
          const bytes = [parseInt(b1, 16), parseInt(b2, 16), parseInt(b3, 16)];
          return new TextDecoder('utf-8').decode(new Uint8Array(bytes));
        });
        // 然后处理单字节字符
        dataStr = dataStr.replace(/\\x([0-9a-fA-F]{2})/g, (match, hex) => {
          return String.fromCharCode(parseInt(hex, 16));
        });
      }
      let obj;
      
      obj = JSON.parse(dataStr);             
      
      // 处理 response 字段
      if (obj.response) {
        // 删除thinking标签
        let cleanResponse = obj.response
          .replace(/\\u003cthink\\u003e/g, '')
          .replace(/\\u003c\/think\\u003e/g, '');
        
        responseText += normalizeEscapes(cleanResponse);
        
        // 更新当前解说区域
        if (currentCommentaryDiv) {
          currentCommentaryDiv.innerHTML = marked.parse(responseText);
        }
        
        // 自动滚动到底部
        responseContent.scrollTop = responseContent.scrollHeight;
      }
    } catch (e) {
      
      console.error('解析错误:', e, '原始数据:', event.data);
    }
  };

  es.onerror = () => {
    es.close();
  };
  
  // 当解说完成时，移除临时div的class
  es.addEventListener('end', () => {
    if (currentCommentaryDiv) {
      currentCommentaryDiv.className = '';
    }
  });
}