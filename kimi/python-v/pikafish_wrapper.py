import subprocess
import threading
import queue
import time
import sys
from typing import Optional, List


class PikafishEngine:
    def __init__(self, exe_path: str = "pikafish.exe"):
        self.exe_path = exe_path
        self.process: Optional[subprocess.Popen] = None
        self.output_queue = queue.Queue()
        self.stdout_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._uci_ready = False  # 新增标志
        self._searching = False

    def start(self):
        """启动 Pikafish 引擎进程"""
        if self.process is not None:
            raise RuntimeError("引擎已在运行")

        self.process = subprocess.Popen(
            [self.exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        self._stop_event.clear()
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stdout_thread.start()

    def _read_stdout(self):
        """从引擎 stdout 读取输出并放入队列"""
        assert self.process is not None
        while not self._stop_event.is_set():
            line = self.process.stdout.readline()
            if line:
                self.output_queue.put(line.strip())
            else:
                break  # EOF

    def _send_command(self, cmd: str):
        """向引擎发送命令"""
        if self.process is None or self.process.stdin.closed:
            raise RuntimeError("引擎未启动或已关闭")
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def get_response(self, timeout: Optional[float] = None) -> List[str]:
        """获取引擎输出，直到队列为空或超时"""
        lines = []
        start_time = time.time()
        while True:
            try:
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    line = self.output_queue.get(timeout=remaining)
                else:
                    line = self.output_queue.get(timeout=0.1)
                lines.append(line)
            except queue.Empty:
                break
        return lines

    def uci(self) -> List[str]:
        """发送 uci 命令并返回响应"""
        self._send_command("uci")
        self._uci_ready = True
        return self.get_response(timeout=5)

    def setoption(self, name: str, value):
        if not self._uci_ready:
            raise RuntimeError("必须先完成 uci() 初始化")
        if self._searching:
            raise RuntimeError("引擎正在搜索中")
        self._send_command(f"setoption name {name} value {value}")

    def position(self, fen: Optional[str] = None, moves: Optional[List[str]] = None):
        """设置局面"""
        cmd = "position"
        if fen is not None:
            cmd += f" fen {fen}"
        else:
            cmd += " startpos"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send_command(cmd)

    def isready(self) -> bool:
        """发送 isready 并等待 readyok"""
        self._send_command("isready")
        start = time.time()
        while time.time() - start < 5:
            try:
                line = self.output_queue.get(timeout=0.1)
                if line == "readyok":
                    return True
            except queue.Empty:
                continue
        return False

    def go(self, **kwargs):
        """
        发送 go 命令，支持参数如：
        - depth=10
        - movetime=1000
        - wtime=10000, btime=10000, winc=1000, binc=1000
        - nodes=100000
        示例: engine.go(depth=15)
        """
        cmd = "go"
        for k, v in kwargs.items():
            cmd += f" {k} {v}"
        self._send_command(cmd)

    def stop(self):
        """发送 stop 命令（用于中断搜索）"""
        self._send_command("stop")

    def quit(self):
        """退出引擎并清理资源"""
        if self.process:
            # self.send_command("quit")
            self._stop_event.set()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return 
    
    """
            执行一次搜索并返回 bestmove。

            参数:
                fen: FEN 字符串（若提供，则忽略 moves）
                moves: 从起始局面开始的走法列表（如 ["e2e4", "e7e5"]）
                depth: 搜索深度
                movetime: 思考时间（毫秒）
                nodes: 搜索节点数
                go_timeout: 等待 bestmove 的最大时间（秒）

            返回:
                bestmove 字符串，如 "e2e4"
            """
    def get_bestmove(
        self,
        fen: Optional[str] = None,
        moves: Optional[List[str]] = None,
        depth: Optional[int] = None,
        movetime: Optional[int] = None,
        nodes: Optional[int] = None,
        go_timeout: float = 10.0
    ) -> str:
    
        if self._searching:
            raise RuntimeError("上一次搜索尚未完成")
        
        # 设置局面
        if fen is not None:
            pos_cmd = f"position fen {fen}"
        else:
            pos_cmd = "position startpos"
            if moves:
                pos_cmd += " moves " + " ".join(moves)
        self._send_command(pos_cmd)

        # 构建 go 命令
        go_cmd = "go"
        if depth is not None:
            go_cmd += f" depth {depth}"
        elif movetime is not None:
            go_cmd += f" movetime {movetime}"
        elif nodes is not None:
            go_cmd += f" nodes {nodes}"
        # 否则使用默认（不推荐）

        self._searching = True
        self._send_command(go_cmd)

        # 等待 bestmove
        start = time.time()
        while time.time() - start < go_timeout:
            try:
                line = self.output_queue.get(timeout=0.1)
                if line.startswith("bestmove"):
                    self._searching = False
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
                    else:
                        raise RuntimeError(f"无效的 bestmove 行: {line}")
            except queue.Empty:
                continue

        # 超时：发送 stop 并再等一次
        self._send_command("stop")
        for _ in range(20):  # 最多再等 2 秒
            try:
                line = self.output_queue.get(timeout=0.1)
                if line.startswith("bestmove"):
                    self._searching = False
                    return line.split()[1]
            except queue.Empty:
                break

        self._searching = False
        raise TimeoutError(f"搜索超时（{go_timeout} 秒），未收到 bestmove")


# 示例用法
if __name__ == "__main__":
    with PikafishEngine("pikafish.exe") as engine:
        # 1. 初始化 UCI
        print("Initializing UCI...")
        uci_resp = engine.uci()
        for line in uci_resp:
            print(line)

        # 第一次搜索：起始局面，深度 12
    move1 = engine.get_bestmove(depth=12)
    print("Best move 1:", move1)

    # 第二次搜索：指定 FEN
    move2 = engine.get_bestmove(
        fen="r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        movetime=1500
    )
    print("Best move 2:", move2)

    # 第三次搜索：从起始局面走几步后
    move3 = engine.get_bestmove(moves=["e2e4", "c7c5", "g1f3", "d7d6"], depth=10)
    print("Best move 3:", move3)