import os

files = os.listdir("./kifu_data/jsonl")

print(files)
with open(f"./kifu_data/jsonl/{files[0]}","r") as f:
    for i in range(100):
        data = f.readline()
        print(data)
        