# 导入必要库
from bs4 import BeautifulSoup
import sqlite3
from sqlite3 import OperationalError
import os

def extract_file_ids_from_html(html_file_path):
    """
    从HTML文件中提取所有符合规则的file_id
    规则：a标签的href属性以"../qipu/"开头，提取后面的UUID部分（如../qipu/xxx -> xxx）
    :param html_file_path: HTML文件的路径（相对/绝对路径均可）
    :return: 去重后的file_id列表，提取失败返回空列表
    """
    # 校验HTML文件是否存在
    if not os.path.exists(html_file_path):
        print(f"错误：HTML文件不存在 -> {html_file_path}")
        return []
    
    # 读取HTML文件内容
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"读取HTML文件失败：{str(e)}")
        return []
    
    # 解析HTML并提取file_id
    soup = BeautifulSoup(html_content, 'html.parser')  # 使用html.parser解析（无需额外安装依赖）
    file_ids = []
    
    # 遍历所有<a>标签
    for a_tag in soup.find_all('a'):
        # 获取a标签的href属性（若不存在则跳过）
        href = a_tag.get('href', '')
        # 筛选href以"../qipu/"开头的标签
        if href.startswith('../qipu/'):
            # 提取file_id（分割href，取"../qipu/"后面的部分）
            file_id = href.split('../qipu/')[-1]
            # 过滤空值并去重（避免重复提取同一file_id）
            if file_id and file_id not in file_ids:
                file_ids.append(file_id)
    
    print(f"成功从HTML中提取到 {len(file_ids)} 个file_id")
    return file_ids


def init_database(db_path="chess.db"):
    """
    初始化SQLite数据库：创建qipu_file_ids表 + 自动更新时间触发器
    :param db_path: 数据库文件路径（默认当前目录下的chess.db）
    :return: 数据库连接对象（conn），失败返回None
    """
    try:
        # 连接数据库（不存在则自动创建）
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. 创建qipu_file_ids表（含主键约束，避免重复插入）
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS qipu_file_ids (
            file_id TEXT PRIMARY KEY,  -- file_id作为主键（自动去重）
            tag INTEGER DEFAULT 0,     -- 处理标志：0-未处理（默认），1-已处理
            update_time TEXT           -- 最后更新时间（由触发器自动维护）
        );
        """
        cursor.execute(create_table_sql)
       
        # 提交事务
        conn.commit()
        print(f"数据库初始化成功（表+触发器）：{db_path}")
        return conn
    
    except OperationalError as e:
        print(f"数据库初始化失败（SQL语法错误）：{str(e)}")
        return None
    except Exception as e:
        print(f"数据库连接失败：{str(e)}")
        return None


def insert_file_ids_to_db(conn, file_ids):
    """
    将提取的file_id批量插入数据库（自动跳过重复值）
    :param conn: 数据库连接对象
    :param file_ids: 待插入的file_id列表
    :return: 成功插入的数量
    """
    if not conn or not file_ids:
        print("插入失败：数据库连接未建立或file_id列表为空")
        return 0
    
    cursor = conn.cursor()
    inserted_count = 0  # 记录成功插入的数量
    
    # 插入时直接通过SQL函数设置当前时间（本地时间）
    # 使用datetime('now', 'localtime')获取当前本地时间，格式为YYYY-MM-DD HH:MM:SS
    insert_sql = """
        INSERT OR IGNORE INTO qipu_file_ids (file_id, update_time) 
        VALUES (?, datetime('now', 'localtime'));
    """
    
    try:
        # 执行批量插入（效率高于单条插入）
        cursor.executemany(insert_sql, [(fid,) for fid in file_ids])
        # 提交事务
        conn.commit()
        # 获取实际插入的行数（executemany返回None，需通过rowcount获取）
        inserted_count = cursor.rowcount
        print(f"批量插入完成：总待插入{len(file_ids)}个，实际插入{inserted_count}个（重复{len(file_ids)-inserted_count}个）")
        return inserted_count
    
    except Exception as e:
        conn.rollback()  # 插入失败时回滚事务
        print(f"插入数据失败：{str(e)}")
        return 0


def main():
    # -------------------------- 请根据实际情况修改以下路径 --------------------------
    HTML_FILE_DIR = "D:/qipu/qipu"  # 你的HTML文件路径
    DB_FILE_PATH = "chess_games.db"          # 数据库文件路径
    # ---------------------------------------------------------------------------------
     # 1. 初始化数据库（创建表和触发器）
    conn = init_database(DB_FILE_PATH)
    if not conn:
        print("流程终止：数据库初始化失败")
        return
    
    # 2. 遍历HTML_FILE_DIR目录下的所有文件
    file_ids = []
    for _, _, files in os.walk(HTML_FILE_DIR):
        for file in files:            
            #获取文件名
            file_name = os.path.splitext(file)[0]
            file_ids.append(file_name)
            
    # 3. 关闭数据库连接
    insert_file_ids_to_db(conn, file_ids)    
    conn.close()
    print("流程结束：数据库连接已关闭")


# 脚本入口（直接运行时执行）
if __name__ == "__main__":
    main()