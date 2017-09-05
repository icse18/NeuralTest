import os
import tensorflow as tf
import argparse
from cgan import cgan
from utils import *
from data_pipeline import *

def parse_args():
    description = "Generative Adversarial Networks for Software Testing"
    parser = argparse.ArgumentParser(description=description)

    # parser.add_argument('--gan_type', type=str, default='CGAN',
    #                     choices=['CGAN'],
    #                     help='The type of GAN',
    #                     required=True)

    parser.add_argument('--epoch', type=int, default=100, help='The number of epoch to run')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated vectors')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory name to save training logs')

    return check_args(parser.parse_args())

def check_args(args):
    # --checkpoint_dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('Number of epochs must be greater than or equal to one...')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one...')

    return args

def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # generate data
    vectors, labels = synthetic_data_generator(predicate_1, 30000, -10000, 10000, 3)
    datasets = prepare_data_sets(vectors, labels, 5000)

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = cgan(sess, epoch=args.epoch, batch_size=args.batch_size, datasets=datasets,
                    checkpoint_dir=args.checkpoint_dir, result_dir=args.result_dir, log_dir=args.log_dir)

        # build graph
        gan.construct_model()

        # show network architecture
        show_all_variables()

        # launch the graph in the session
        gan.train()
        print("[*] Training completed!")

if __name__ == '__main__':
    main()
