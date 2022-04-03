import argparse

from make_shards import MakeShards

if __name__ == '__main__':

    """
    Adapted from https://github.com/mwalmsley/zoobot/blob/main/decals_dr5_to_shards.py
    
    See shards_*.sh for some suitable shell commands
    """

    shard_types = ['gz2', 'gz2_partial', 'decals', 'decals_partial']

    parser = argparse.ArgumentParser(description='Make shards')

    # you should have already made these catalogs for your dataset
    parser.add_argument('--labelled-catalog', dest='labelled_catalog_loc', type=str,
                    help='Path to csv catalog of previous labels and file_loc, for shards')
    parser.add_argument('--unlabelled-catalog', dest='unlabelled_catalog_loc', type=str, default='',
                help="""Path to csv catalog of previous labels and file_loc, for shards. 
                Optional - skip (recommended) if all galaxies are labelled.""")

    parser.add_argument('--eval-size', dest='eval_size', type=int,
        help='Split labelled galaxies into train/test, with this many test galaxies (e.g. 5000)')

    # Write catalog to shards (tfrecords as catalog chunks) here
    parser.add_argument('--shard-type', dest='shard_type', type=str,
                    choices=shard_types,
                    help=f'Choose from {shard_types}')
    parser.add_argument('--shard-dir', dest='shard_dir', type=str,
                    help='Directory into which to place shard directory')
    parser.add_argument('--max-unlabelled', dest='max_unlabelled', type=int,
                    help='Max unlabelled galaxies (for debugging/speed')
    parser.add_argument('--max-labelled', dest='max_labelled', type=int,
                    help='Max labelled galaxies (for debugging/speed')
    parser.add_argument('--img-size', dest='img_size', type=int,
                    help='Size at which to save images (before any augmentations). 300 for DECaLS paper.')

    args: dict = parser.parse_args()

    ms = MakeShards()
    ms.set_shard_type(args.shard_type)
    ms.set_catalogs(labelled_catalog_loc=args.labelled_catalog_loc,
                    unlabelled_catalog_loc=args.unlabelled_catalog_loc,
                    max_labelled=args.max_labelled,
                    max_unlabelled=args.max_unlabelled)
    ms.make_shards(shard_dir=args.shard_dir,
                   img_size=args.img_size,
                   eval_size=args.eval_size)

