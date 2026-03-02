import os
import sys
import random
import cshogi
import usi_server

class RandomEngine(usi_server.USIEngine):
    def __init__(self, engine_name, author_name):
        super().__init__(engine_name, author_name)
    
    def think(self):
        moves = list(self.board.legal_moves)

        if not moves:
            self.send("bestmove resign")
            self.thinking = False
            return

        # ランダム
        move = random.choice(moves)

        self.send(f"bestmove {cshogi.move_to_usi(move)}")
        self.thinking = False

if __name__ == "__main__":
    engine_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    engine = RandomEngine(engine_name, "withrice")
    engine.loop()