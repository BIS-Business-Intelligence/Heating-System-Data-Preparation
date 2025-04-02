import os

def create_directory(path: str):
    if os.path.exists(path):
        print(f"Directory '{path}' already exists.")
        return

    print(f"Created directory '{path}'.")
    os.mkdir(path)