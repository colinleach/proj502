from pathlib import Path
import logging
import pandas as pd

def gz2_partial_catalog():
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

    data = pg.run_select(sql)

    headers = ['id_str',
               'file_loc',
               'smooth-or-featured_smooth',
               'smooth-or-featured_featured-or-disk']

    df = pd.DataFrame(data, columns=headers)
    df['id_str'] = df['id_str'].str.rstrip()

    df.to_csv('gz2_partial_pairs.csv', index=False)


gz2_pairs = {
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

headers = []
for k in gz2_pairs:
    for v in gz2_pairs[k]:
        headers.append(k + v)