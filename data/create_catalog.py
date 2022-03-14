from pathlib import Path
import logging
import pandas as pd

from utils import read_params
from db import DB
from zoobot.label_metadata import gz2_pairs, decals_pairs, decals_partial_pairs


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

        headers = ['id_str', 'file_loc', ]
        for k in gz2_pairs:
            for v in gz2_pairs[k]:
                headers.append(k + v)

        df = pd.DataFrame(data, columns=headers)
        df['id_str'] = df['id_str'].str.rstrip()

        catalog_file = self.catalog_dir / 'gz2_pairs.csv'
        df.to_csv(catalog_file, index=False)
        logging.info(f"Wrote {df.shape[0]} data rows to {catalog_file}")

    def decals_catalog(self):
        """

        """

        sql = """
            SELECT  
                iauname, 
                path as file_loc,
                smooth_or_featured_smooth,
                smooth_or_featured_featured_or_disk,
                smooth_or_featured_artifact,
                disk_edge_on_yes,
                disk_edge_on_no,
                has_spiral_arms_yes,
                has_spiral_arms_no,
                bar_strong,
                bar_weak,
                bar_no,
                bulge_size_dominant,
                bulge_size_large,
                bulge_size_moderate,
                bulge_size_small,
                bulge_size_none,
                how_rounded_round,
                how_rounded_in_between,
                how_rounded_cigar_shaped,
                edge_on_bulge_boxy,
                edge_on_bulge_none,
                edge_on_bulge_rounded,
                spiral_winding_tight,
                spiral_winding_medium,
                spiral_winding_loose,
                spiral_arm_count_1,
                spiral_arm_count_2,
                spiral_arm_count_3,
                spiral_arm_count_4,
                spiral_arm_count_more_than_4,
                spiral_arm_count_cant_tell,
                merging_none,
                merging_minor_disturbance,
                merging_major_disturbance,
                merging_merger                          
            FROM 
                decalsdata d, img
            WHERE 
                filetype = 'png' 
            AND 
                d.iauname = img.id_str
            """

        data = self.pg.run_select(sql)

        headers = ['id_str', 'file_loc', ]
        for k in decals_pairs:
            for v in decals_pairs[k]:
                headers.append(k + v)

        df = pd.DataFrame(data, columns=headers)
        df['id_str'] = df['id_str'].str.rstrip()

        catalog_file = self.catalog_dir / 'decals_pairs.csv'
        df.to_csv(catalog_file, index=False)
        logging.info(f"Wrote {df.shape[0]} data rows to {catalog_file}")


if __name__ == '__main__':
    cat = Catalog()
    # cat.gz2_partial_catalog()
    # cat.gz2_catalog()
    cat.decals_catalog()
