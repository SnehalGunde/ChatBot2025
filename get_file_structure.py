import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level + '├── '
        print(indent + os.path.basename(root) + '/')
        subindent = '│   ' * (level + 1)
        for f in files:
            print(subindent + f)

list_files(".")
