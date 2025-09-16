import requests

OLLAMA_URL = "https://openai.qzz8io.qzz.io/api/generate"
payload = {
    "model": "gpt-oss:latest",  # 请根据你本地已拉取的模型名称修改
    "prompt": "请用简洁中文介绍中国象棋的基本规则。"
}

print("发送请求到 Ollama ...")
try:
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=30) as r:
        print("状态码:", r.status_code)
        if r.status_code != 200:
            print("Ollama 返回错误:", r.text)
        else:
            print("Ollama 输出：")
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    print(line)
except Exception as e:
    print("请求异常:", e)