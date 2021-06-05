import os


PROJECT_ROOT = '.'


def out(local_path):
    return os.path.join(PROJECT_ROOT, 'out', local_path)
