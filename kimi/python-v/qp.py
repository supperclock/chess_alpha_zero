from bs4 import BeautifulSoup

# 您的网页源代码（已截断，但包含关键信息）
html_source = """
<!DOCTYPE html><html lang="zh-hans" dir="ltr"><head>...</head><body class="html not-front...">
  ...
    <div class="row">
        <section class="col-sm-12">...<h1 class="page-header">李洛如 先负 彭诗文-2025第十九届世界象棋锦标赛</h1>...
          <div class="navbar-collapse collapse">...</div>
          <div class="navbar-collapse collapse">...</div>
      <div class="col-sm-3">	  <ul id="moves_text" class="content qipu-content"><li class="start"><span class="move active" name="0">==== 棋局开始 ====</span></li><li class="round"><span class="roundnum">1.</span><span class="move" name="1">炮二平五</span><span class="move" name="2">炮8平5</span></li><li class="round"><span class="roundnum">2.</span><span class="move" name="3">马二进三</span><span class="move" name="4">马8进7</span></li><li class="round"><span class="roundnum">3.</span><span class="move" name="5">车一平二</span><span class="move" name="6">卒7进1</span></li><li class="round"><span class="roundnum">4.</span><span class="move" name="7">炮八平六</span><span class="move" name="8">马2进3</span></li><li class="round"><span class="roundnum">5.</span><span class="move" name="9">马八进七</span><span class="move" name="10">车1平2</span></li><li class="round"><span class="roundnum">6.</span><span class="move" name="11">车九平八</span><span class="move" name="12">车9进1</span></li><li class="round"><span class="roundnum">7.</span><span class="move" name="13">兵七进一</span><span class="move" name="14">车9平4</span></li><li class="round"><span class="roundnum">8.</span><span class="move" name="15">仕四进五</span><span class="move" name="16">炮2进4</span></li><li class="round"><span class="roundnum">9.</span><span class="move" name="17">相七进九</span><span class="move" name="18">车4进3</span></li><li class="round"><span class="roundnum">10.</span><span class="move" name="19">车二进四</span><span class="move" name="20">马7进6</span></li><li class="round"><span class="roundnum">11.</span><span class="move" name="21">兵三进一</span><span class="move" name="22">马6进7</span></li><li class="round"><span class="roundnum">12.</span><span class="move" name="23">兵三进一</span><span class="move" name="24">车4平7</span></li><li class="round"><span class="roundnum">13.</span><span class="move" name="25">车二平六</span><span class="move" name="26">士6进5</span></li><li class="round"><span class="roundnum">14.</span><span class="move" name="27">车六进二</span><span class="move" name="28">炮5平7</span></li><li class="round"><span class="roundnum">15.</span><span class="move" name="29">车六平七</span><span class="move" name="30">象7进5</span></li><li class="round"><span class="roundnum">16.</span><span class="move" name="31">炮五平四</span><span class="move" name="32">炮2平3</span></li><li class="round"><span class="roundnum">17.</span><span class="move" name="33">车八进九</span><span class="move" name="34">炮3退3</span></li><li class="round"><span class="roundnum">18.</span><span class="move" name="35">车八退三</span><span class="move" name="36">炮3进4</span></li><li class="round"><span class="roundnum">19.</span><span class="move" name="37">炮四平七</span><span class="move" name="38">马7退6</span></li><li class="round"><span class="roundnum">20.</span><span class="move" name="39">马三进四</span><span class="move" name="40">车7进5</span></li><li class="round"><span class="roundnum">21.</span><span class="move" name="41">仕五退四</span><span class="move" name="42">炮7平6</span></li><li class="round"><span class="roundnum">22.</span><span class="move" name="43">炮七进五</span><span class="move" name="44">炮6进3</span></li><li class="round"><span class="roundnum">23.</span><span class="move" name="45">车八平五</span><span class="move" name="46">炮6平8</span></li><li class="round"><span class="roundnum">24.</span><span class="move" name="47">炮六平二</span><span class="move" name="48">车7退2</span></li><li class="round"><span class="roundnum">25.</span><span class="move" name="49">炮二退一</span><span class="move" name="50">马6进7</span></li><li class="round"><span class="roundnum">26.</span><span class="move" name="51">车五平三</span><span class="move" name="52">车7平5</span></li><li class="round"><span class="roundnum">27.</span><span class="move" name="53">炮二平五</span><span class="move" name="54">炮8进4</span></li></ul>	  </div>
      ...
          <div class="col-sm-6">
<div class="panel panel-default tuwenqipu">  <div class="panel-heading">图文格式象棋棋谱</div>  <div class="panel-body">  <div class="qipu-basic">
    ...
    <div id="qipu-moves-iccs">h2e2h7e7h0g2h9g7i0h0g6g5b2d2b9c7b0c2a9b9a0b0i9i8c3c4i8d8f0e1b7b3c0a2d8d5h0h4g7f5g3g4f5g3g4g5d5g5h4d4f9e8d4d6e7g7d6c6g9e7e2f2b3c3b0b9c3c6b9b6c6c2f2c2g3f5g2f4g5g0e1f0g7f7c2c7f7f4b6e6f4h4d2h2g0g2h2h1f5g3e6g6g2e2h1e1h4h0</div>	    <div id="qipu-init-fen"></div>		</div>...
"""

# 使用 BeautifulSoup 解析 HTML 源码
soup = BeautifulSoup(html_source, 'html.parser')

# ---
## 1. 提取文字描述棋谱（如：炮二平五）
# ---
# 棋谱位于 ID 为 "moves_text" 的 ul 列表下
moves_list = []
move_container = soup.find('ul', id='moves_text')

if move_container:
    # 查找所有 class 为 'round' 的 <li> 标签，它们包含了完整的红黑双方一步棋
    round_elements = move_container.find_all('li', class_='round')
    
    for round_li in round_elements:
        round_num = round_li.find('span', class_='roundnum').get_text(strip=True).replace('.', '')
        # 查找该回合下的所有 'move' span
        moves = round_li.find_all('span', class_='move')
        
        red_move = moves[0].get_text(strip=True) if len(moves) > 0 else "N/A"
        black_move = moves[1].get_text(strip=True) if len(moves) > 1 else ""
        
        # 格式化为：回合数. 红方招法 黑方招法
        moves_list.append(f"{round_num}. {red_move} {black_move}")

print("--- 提取的文字描述棋谱 (回合列表) ---")
for move in moves_list:
    print(move)

# ---
## 2. 提取 ICCS 格式棋谱
# ---
# 棋谱的 ICCS (International Chinese Chess System) 格式位于 ID 为 "qipu-moves-iccs" 的 div 中
iccs_container = soup.find('div', id='qipu-moves-iccs')

iccs_moves = "未找到ICCS棋谱"
if iccs_container:
    # 直接获取 div 标签的文本内容，即为 ICCS 棋谱字符串
    iccs_moves = iccs_container.get_text(strip=True)

print("\n--- 提取的 ICCS 格式棋谱 (原始字符串) ---")
print(iccs_moves)


# ---
## 3. 将 ICCS 棋谱格式化为列表
# ---
# ICCS 棋谱是每4个字符代表一步棋（例如：h2e2）
formatted_iccs_list = []
if iccs_moves != "未找到ICCS棋谱":
    # 遍历字符串，每次截取4个字符
    for i in range(0, len(iccs_moves), 4):
        move = iccs_moves[i:i+4]
        if move:
            formatted_iccs_list.append(move)

print("\n--- 格式化后的 ICCS 棋谱 (列表) ---")
print(formatted_iccs_list)