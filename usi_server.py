import os
import sys
import cshogi
import threading
from dotenv import load_dotenv

load_dotenv('../../pram/.env') # 実行ファイルからの相対パス
AUTHOR = os.getenv('AUTHOR')

class USIEngine:
    def __init__(self, engine_name):
        self.engine_name = engine_name
        self.author_name = AUTHOR
        self.board = cshogi.Board()
        self.thinking = False
        self.stop_flag = False
    
    def send(self, msg):
        print(msg, flush=True)
    
    def info(self, depth, time, cp, pv):
        self.send(f"info depth {depth} time {time} nodes {self.nodes} score cp {cp} pv {' '.join(pv)}")
    
    def loop(self):
        while True:
            line = sys.stdin.readline().strip()

            if not line:
                continue

            if line == "usi":
                self.send(f"id name {self.engine_name}")
                self.send(f"id author {self.author_name}")
                self.send("usiok")

            elif line == "isready":
                self.send("readyok")

            elif line.startswith("position"):
                self.cmd_position(line)

            elif line.startswith("go"):
                self.cmd_go()

            elif line == "stop":
                self.stop_flag = True

            elif line == "quit":
                break
        
    
    def cmd_position(self, line):
        tokens = line.split()

        if "startpos" in tokens:
            self.board.reset()

        if "moves" in tokens:
            idx = tokens.index("moves")
            moves = tokens[idx + 1:]

            for mv in moves:
                self.board.push_usi(mv)
    
    def cmd_go(self):
        if self.thinking:
            return

        self.stop_flag = False
        self.thinking = True

        t = threading.Thread(target=self.think)
        t.start()

    def think(self):
        self.send("bestmove resign")
        self.thinking = False
        return
