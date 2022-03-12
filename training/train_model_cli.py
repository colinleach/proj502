import argparse

if __name__ == '__main__':
    """
    DECaLS debugging (make the shards first with create_shards.py):
      python train_model.py --experiment-dir results/decals_debug --shard-img-size 32 --resize-size 224 --train-dir data/decals/shards/decals_debug/train_shards --eval-dir data/decals/shards/decals_debug/eval_shards --epochs 2 --batch-size 8

    DECaLS full:
      python train_model.py --experiment-dir results/decals_debug --shard-img-size 300 --train-dir /raid/scratch/walml/galaxy_zoo/decals/tfrecords/all_2p5_unfiltered_retired/train_shards --eval-dir /raid/scratch/walml/galaxy_zoo/decals/tfrecords/all_2p5_unfiltered_retired/eval_shards --epochs 200 --batch-size 256 --resize-size 224
    New features: add --distributed for multi-gpu, --wandb for weights&biases metric tracking, --color for color (does not perform better)

    GZ2 debugging:
      python train_model.py --experiment-dir results/gz2_debug --shard-img-size 300 --train-dir data/gz2/shards/all_sim_2p5_unfiltered_300/train_shards --eval-dir data/gz2/shards/all_sim_2p5_unfiltered_300/train_shards --epochs 1 --batch-size 8 --resize-size 128


    """

    pair_types = ['gz2', 'gz2_partial', 'decals', 'decals_partial']

    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-type', dest='pair_type', type=str,
                    choices=pair_types,
                    help=f'Choose from {pair_types}')
    parser.add_argument('--experiment-dir', dest='save_dir', type=str)
    parser.add_argument('--shard-img-size', dest='shard_img_size', type=int, default=300)
    parser.add_argument('--resize-size', dest='resize_size', type=int, default=224)
    parser.add_argument('--train-dir', dest='train_records_dir', type=str)
    parser.add_argument('--eval-dir', dest='eval_records_dir', type=str)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--color', default=False, action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--eager', default=False, action='store_true',
        help='Use TensorFlow eager mode. Great for debugging, but significantly slower to train.'),
    parser.add_argument('--test-time-augs', dest='always_augment', default=False, action='store_true',
        help='Zoobot includes keras.preprocessing augmentation layers. \
        These only augment (rotate/flip/etc) at train time by default. \
        They can be enabled at test time as well, which gives better uncertainties (by increasing variance between forward passes) \
        but may be unexpected and mess with e.g. GradCAM techniques.'),
    parser.add_argument('--dropout-rate', dest='dropout_rate', default=0.2, type=float)
    args = parser.parse_args()
