from pathlib import Path
import logging
import pandas as pd

from utils import read_params
from db import DB


class Catalog:

    def __init__(self):
        params = read_params()
        self.catalog_dir = Path(params['dataroot']) / params['catalogs']
        self.pg = DB()
        logging.basicConfig(
            filename='gzdata.log',
            # filemode='w',
            format='%(asctime)s %(levelname)s:%(message)s',
            level=logging.INFO
        )

    def gz2_partial_catalog(self):
        """

        """

        sql = """
            SELECT  
                id_str, 
                path as file_loc, 
                t01_smooth_or_features_a01_smooth_count,
                t01_smooth_or_features_a02_features_or_disk_count
            FROM 
                gz2data g, img
            WHERE 
                filetype = 'png' 
            AND 
                g.dr7objid = img.dr7id
            """

        data = self.pg.run_select(sql)

        headers = ['id_str',
                   'file_loc',
                   'smooth-or-featured_smooth',
                   'smooth-or-featured_featured-or-disk',]

        df = pd.DataFrame(data, columns=headers)
        df['id_str'] = df['id_str'].str.rstrip()

        catalog_file = self.catalog_dir / 'gz2_partial_pairs.csv'
        df.to_csv(catalog_file, index=False)
        logging.info(f"Wrote {df.shape[0]} data rows to {catalog_file}")

    def gz2_catalog(self):
        """

        """

        sql = """
            SELECT  
                id_str, 
                path as file_loc, 
                t01_smooth_or_features_a01_smooth_count,
                t01_smooth_or_features_a02_features_or_disk_count,
                t02_edgeon_a04_yes_count,
                t02_edgeon_a05_no_count,
                t04_spiral_a08_spiral_count,
                t04_spiral_a09_no_spiral_count,
                t03_bar_a06_bar_count,
                t03_bar_a07_no_bar_count,
                t05_bulge_prominence_a13_dominant_count,
                t05_bulge_prominence_a12_obvious_count,
                t05_bulge_prominence_a11_just_noticeable_count,
                t05_bulge_prominence_a10_no_bulge_count,
                t06_odd_a14_yes_count,
                t06_odd_a15_no_count,
                t07_rounded_a16_completely_round_count,
                t07_rounded_a17_in_between_count,
                t07_rounded_a18_cigar_shaped_count,
                t09_bulge_shape_a25_rounded_count,
                t09_bulge_shape_a26_boxy_count,
                t09_bulge_shape_a27_no_bulge_count,
                t10_arms_winding_a28_tight_count,
                t10_arms_winding_a29_medium_count,
                t10_arms_winding_a30_loose_count,
                t11_arms_number_a31_1_count,
                t11_arms_number_a32_2_count,
                t11_arms_number_a33_3_count,
                t11_arms_number_a34_4_count,
                t11_arms_number_a36_more_than_4_count,
                t11_arms_number_a37_cant_tell_count
            FROM 
                gz2data g, img
            WHERE 
                filetype = 'png' 
            AND 
                g.dr7objid = img.dr7id
            """

        data = self.pg.run_select(sql)

        gz2_pairs = {  # copied from label_metadata.py
            'smooth-or-featured': ['_smooth', '_featured-or-disk'],
            'disk-edge-on': ['_yes', '_no'],
            'has-spiral-arms': ['_yes', '_no'],
            'bar': ['_yes', '_no'],
            'bulge-size': ['_dominant', '_obvious', '_just-noticeable', '_no'],
            'something-odd': ['_yes', '_no'],
            'how-rounded': ['_round', '_in-between', '_cigar'],
            'bulge-shape': ['_round', '_boxy', '_no-bulge'],
            'spiral-winding': ['_tight', '_medium', '_loose'],
            'spiral-count': ['_1', '_2', '_3', '_4', '_more-than-4', '_cant-tell']
        }
        headers = ['id_str', 'file_loc',]
        for k in gz2_pairs:
            for v in gz2_pairs[k]:
                headers.append(k + v)

        df = pd.DataFrame(data, columns=headers)
        df['id_str'] = df['id_str'].str.rstrip()

        catalog_file = self.catalog_dir / 'gz2_pairs.csv'
        df.to_csv(catalog_file, index=False)
        logging.info(f"Wrote {df.shape[0]} data rows to {catalog_file}")

if __name__ == '__main__':
    cat = Catalog()
    cat.gz2_partial_catalog()
    cat.gz2_catalog()
