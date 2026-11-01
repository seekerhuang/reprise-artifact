import os
from pathlib import Path


def configure():
    root = Path(__file__).parent / 'lib'
    if not (root / 'libdgl.so').is_file():
        raise RuntimeError('Run python build_extensions.py first')
    os.environ['DGLBACKEND'] = 'pytorch'
    os.environ['DGL_LIBRARY_PATH'] = str(root.resolve())
