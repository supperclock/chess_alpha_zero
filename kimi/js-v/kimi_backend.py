from flask import Flask, request, jsonify, Response, send_from_directory, stream_with_context
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# 创建static目录
static_dir = 'static'
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

OLLAMA_URL = "https://openai.qzz8io.qzz.io/api/generate"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """处理静态文件请求"""
    # 首先尝试从static目录获取文件
    if os.path.exists(os.path.join(static_dir, filename)):
        return send_from_directory(static_dir, filename)
    # 如果static目录中没有，尝试从根目录获取
    elif os.path.exists(filename):
        return send_from_directory('.', filename)
    else:
        return "File not found", 404

@app.route('/api/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        model = request.args.get("model", "qwen3:latest")
        prompt = request.args.get("prompt", "")
    else:  # POST
        data = request.get_json()
        model = data.get("model", "gpt-oss:latest")
        prompt = data.get("prompt", "")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }
    print(f"[后端] 收到请求: {payload}", flush=True)

    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=600,
        )

        def generate_stream():
            try:
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # print(f"[后端] Ollama输出: {line}", flush=True)
                    yield f"data: {line}\n\n"
            except GeneratorExit:
                print("[后端] 客户端断开连接", flush=True)
            finally:
                r.close()

        return Response(
            stream_with_context(generate_stream()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'text/event-stream; charset=utf-8'
            }
        )

    except Exception as e:
        print(f"[后端] 调用Ollama异常: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
