import os
import sys
import argparse
sys.path.append(os.getcwd())
from wae import WAEGAN, MMDWAE
from noise_sampler import NoiseSampler
from utils.config import dump_config
from mnist.autoencoder import ConvAutoEncoder, FCAutoEncoder
from mnist.discriminator import Discriminator
from mnist.custom_image_sampler import ImageSampler
from kernels import rbf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=300)
    parser.add_argument('--latent_dim', '-ld', type=int, default=2)
    parser.add_argument('--lambda', '-l', type=float, default=0.1, dest='lambda_')
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=1)
    parser.add_argument('--model_dir', '-md', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--noise_mode', '-nm', type=str, default="normal")
    parser.add_argument('--autoencoder', '-ae', type=str, default="fc", choices=['fc', 'conv'])
    parser.add_argument('--base', '-b', type=str, default="gan", choices=['gan', 'mmd'])

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    dump_config(os.path.join(args.result_dir, 'config.csv'), args)

    noise_sampler = NoiseSampler(args.noise_mode)

    if args.autoencoder == 'fc':
        autoencoder = FCAutoEncoder((784, ),
                                    latent_dim=args.latent_dim,
                                    last_activation='tanh',
                                    is_training=True)
        image_sampler = ImageSampler(args.batch_size,
                                     shuffle=True,
                                     is_training=True,
                                     is_vectorize=True)

    elif args.autoencoder == 'conv':
        autoencoder = ConvAutoEncoder((28, 28, 1),
                                      latent_dim=args.latent_dim,
                                      last_activation='tanh',
                                      is_training=True)
        image_sampler = ImageSampler(args.batch_size,
                                     shuffle=True,
                                     is_training=True,
                                     is_vectorize=False)
    else:
        raise NotImplementedError

    discriminator = Discriminator(is_training=True)

    if args.base == 'gan':
        wae = WAEGAN(autoencoder,
                     discriminator,
                     lambda_=args.lambda_,
                     is_training=True)
    elif args.base == 'mmd':
        wae = MMDWAE(autoencoder,
                     rbf(),
                     lambda_=args.lambda_,
                     is_training=True)
    else:
        raise NotImplementedError

    wae.fit_generator(image_sampler,
                      noise_sampler,
                      nb_epoch=args.nb_epoch,
                      save_steps=args.save_steps,
                      visualize_steps=args.visualize_steps,
                      result_dir=args.result_dir,
                      model_dir=args.model_dir)


if __name__ == '__main__':
    main()