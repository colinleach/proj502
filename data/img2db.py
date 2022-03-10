from pathlib import Path
import logging

from db import DB
from utils import read_params

logging.basicConfig(filename='gzoo.log', format='%(asctime)s %(message)s', level=logging.DEBUG)

pg = DB()
params = read_params()

dataroot = Path(params['dataroot'])
jpg_path = dataroot / params['sdssdr7']
png_path = dataroot / params['sdsspng']
decals_path = dataroot / params['decalsdr5']

def insert_rec(id_str, dr7id, path, size, survey, filetype):
    if dr7id is None:
        dr7id = 'NULL'
    sql = f"""
        insert into img
        (id_str, dr7id, path, size, survey, filetype)
        values ('{id_str}', {dr7id}, '{path}', {size}, '{survey}', '{filetype}')
        on conflict do nothing
        """
    pg.run_admin(sql)

def gz2(path, stem):
    count = 0
    for f in path.rglob(f'*.{stem}'):
        name = f.stem
        insert_rec(name, int(name), f, 424, 'SDSSDR7', stem)
#         print(f)
        count += 1
        if count > 5: break
        if count % 100000 == 0:
            logging.info(f'gzoo.img, {count} records inserted)')

    logging.info(f'gzoo.img, finished, {count} records inserted)')

gz2(jpg_path, 'jpg')

def decals(path, stem):
    count = 0
    for f in path.rglob(f'*.{stem}'):
        name = f.stem
        insert_rec(name, None, f, 424, 'DECaLS_5', stem)
#         print(f)
        count += 1
#         if count > 5: break
        if count % 100000 == 0:
            logging.info(f'gzoo.img, {count} records inserted)')


    logging.info(f'gzoo.img, finished, {count} records inserted)')

decals(decals_path, 'png')







