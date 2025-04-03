import os

def create_directory(path: str):
    """
    Creates a directory if it does not exist.
    """
    if os.path.exists(path):
        print(f"Directory '{path}' already exists.")
        return

    print(f"Created directory '{path}'.")
    os.mkdir(path)