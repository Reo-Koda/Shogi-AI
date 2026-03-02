import sys
import shogi
import threading
import time
import random


class USIEngine:
    def __init__(self):
        self.board = shogi.Board()
        self.thinking = False
        self.stop_flag = False

    def send(self, msg):
        print(msg)
        sys.stdout.flush()

    def loop(self):
        while True:
            line = sys.stdin.readline().strip()

            if not line:
                continue

            if line == "usi":
                self.cmd_usi()

            elif line == "isready":
                self.cmd_isready()

            elif line.startswith("position"):
                self.cmd_position(line)

            elif line.startswith("go"):
                self.cmd_go()

            elif line == "stop":
                self.stop_flag = True

            elif line == "quit":
                break

    # ----------------------

    def cmd_usi(self):
        self.send("id name PythonUSI")
        self.send("id author You")
        self.send("usiok")

    def cmd_isready(self):
        self.send("readyok")

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
        legal = list(self.board.legal_moves)

        if not legal:
            self.send("bestmove resign")
            self.thinking = False
            return

        # 仮：ランダム
        move = random.choice(legal)

        # 擬似思考時間
        for _ in range(10):
            if self.stop_flag:
                break
            time.sleep(0.1)

        self.send(f"bestmove {move.usi()}")
        self.thinking = False


if __name__ == "__main__":
    engine = USIEngine()
    engine.loop()
