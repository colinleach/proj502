"""
Convert jpg cutouts (from SDSS) to png images (used by zoobot)

This version is single-threaded and quite slow: several hours for 300k files.
Hopefully it only needs to run once.
"""

from pathlib import Path
from PIL import Image, UnidentifiedImageError
from utils import read_params

params = read_params()
jpg_path = Path(params['dataroot']) / params['sdssdr7']
png_path = Path(params['dataroot']) / params['sdsspng']
print(jpg_path)

for f in jpg_path.glob('*.jpg'):
    out_path = (png_path / f.name).with_suffix('.png')
    if out_path.is_file():
        continue
    try:
        im = Image.open(f)
        im.save(out_path)
    except UnidentifiedImageError:
        pass
