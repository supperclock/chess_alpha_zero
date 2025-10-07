import logging

logging.basicConfig(
    filename='app.log',           # 日志文件名
    filemode='a',                 # 写入模式：'a' 追加，'w' 覆盖
    level=logging.INFO,           # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info('这是一个测试日志')
