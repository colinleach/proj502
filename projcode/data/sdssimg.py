import requests
from pathlib import Path
import logging

from db import DB


class SDSS:
    """
    Class to fetch cutouts from the SDSS DR14 REST service
    """

    def __init__(self):
        logging.basicConfig(
            filename='gzoo.log',
            # filemode='w',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO
        )
        self.pg = DB()
        params = self.pg.read_params()
        self.img_path = Path(params['dataroot']) / params['sdssdr7']
        # self.datafile = params['datafile']
        self.sdss_url = "http://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg"
        self.coords = None

    def db_read(self, limit: int = None) -> None:
        """

        :param limit: Max files to fetch
        :return: None
        """

        sql = """
            SELECT dr7objid, ra, dec 
            FROM gz2data 
            WHERE sample = 'original'
            ORDER BY dr7objid 
            """
        if limit is not None:
            sql += f"LIMIT {limit}"
        self.coords = self.pg.run_select(sql)

    def get_one_jpeg(self, dr7id: int, ra: float, dec: float, jpeg_size: int = 424, scale: float = 0.1) -> int:
        """

        """

        img_fname = self.img_path / Path(f"{dr7id}.jpg")
        if img_fname.is_file():
            return 0
        img_url = self.sdss_url + f"?ra={ra:.5f}&dec={dec:.5f}&width={jpeg_size}&height={jpeg_size}&scale={scale}"
        img_data = requests.get(img_url).content
        with open(img_fname, 'wb') as handler:
            handler.write(img_data)
        return 1

    def target_loop(self, count: int = None, jpeg_size: int = 424) -> None:

        if self.coords is None:
            self.db_read(count)

        count = 0
        for gal in self.coords:
            dr7objid, ra, dec = gal
            downloaded = self.get_one_jpeg(dr7objid, ra, dec, jpeg_size)
            count += downloaded

        logging.info(f"{count} images written to {self.img_path}")

if __name__ == "__main__":
    sdss = SDSS()
    sdss.target_loop()