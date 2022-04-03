"""
Convert jpg cutouts (from SDSS) to png images (used by zoobot)

This version is single-threaded and quite slow: several hours for 300k files.
Hopefully it only needs to run once.
"""

from pathlib import Path
from PIL import Image
from utils import read_params

params = read_params()
jpg_path = Path(params['sdssdr7'])
png_path = Path(params['sdsspng'])


for f in jpg_path.glob('*.jpg'):
    im = Image.open(f)
    out_path = (png_path / f.name).with_suffix('.png')
    im.save(out_path)
