import subprocess
import threading
import queue
import logging
from typing import Optional, List, Dict, Callable

class PikafishEngine:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.process: Optional[subprocess.Popen] = None
        self.command_queue = queue.Queue()
        self.running = False
        self.listeners: Dict[str, List[Callable]] = {
            'bestmove': [],
            'info': []
        }
        
    def start(self):
        """启动引擎进程"""
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.running = True
            
            # 启动读写线程
            threading.Thread(target=self._read_output, daemon=True).start()
            threading.Thread(target=self._write_input, daemon=True).start()
            
            # 初始化引擎
            self.send_command('uci')
            self.send_command('setoption name Threads value 4')
            self.send_command('setoption name Hash value 128')
            self.send_command('isready')
            
        except Exception as e:
            logging.error(f"Failed to start engine: {e}")
            raise
    
    def stop(self):
        """停止引擎"""
        if self.process:
            self.running = False
            self.send_command('quit')
            self.process.terminate()
            self.process = None
            
    def send_command(self, command: str):
        """发送UCI命令到引擎"""
        self.command_queue.put(command)
        
    def _read_output(self):
        """读取引擎输出的线程"""
        while self.running and self.process:
            try:
                line = self.process.stdout.readline().strip()
                if line:
                    self._process_output(line)
            except Exception as e:
                logging.error(f"Error reading engine output: {e}")
                break
                
    def _write_input(self):
        """向引擎写入命令的线程"""
        while self.running:
            try:
                command = self.command_queue.get()
                if command:
                    self.process.stdin.write(f"{command}\n")
                    self.process.stdin.flush()
            except Exception as e:
                logging.error(f"Error writing to engine: {e}")
                break
                
    def _process_output(self, line: str):
        """处理引擎输出"""
        if line.startswith('bestmove'):
            parts = line.split()
            bestmove = parts[1]
            ponder = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
            for callback in self.listeners['bestmove']:
                callback(bestmove, ponder)
        elif line.startswith('info'):
            info = self._parse_info(line)
            for callback in self.listeners['info']:
                callback(info)

    def set_position(self, fen: str, moves: List[str] = None):
        """设置棋局位置"""
        command = f"position fen {fen}"
        if moves:
            command += f" moves {' '.join(moves)}"
        self.send_command(command)

    def start_search(self, time_control: Dict[str, int]):
        """开始搜索"""
        command = "go"
        if 'wtime' in time_control:
            command += f" wtime {time_control['wtime']}"
        if 'btime' in time_control:
            command += f" btime {time_control['btime']}"
        if 'winc' in time_control:
            command += f" winc {time_control['winc']}"
        if 'binc' in time_control:
            command += f" binc {time_control['binc']}"
        if 'depth' in time_control:
            command += f" depth {time_control['depth']}"
        self.send_command(command)

    def stop_search(self):
        """停止搜索"""
        self.send_command('stop')

    def add_listener(self, event: str, callback: Callable):
        """添加事件监听器"""
        if event in self.listeners:
            self.listeners[event].append(callback)