import tensorflow as tf
from blocks import conv_block
from layers import dense, flatten, reshape
from models import AutoEncoder as AE


class ConvAutoEncoder(AE):
    def __init__(self, input_shape,
                 latent_dim,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 is_dropout=False,
                 is_training=True):
        super().__init__(input_shape,
                         latent_dim,
                         last_activation,
                         normalization,
                         upsampling,
                         is_dropout,
                         is_training)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            with tf.variable_scope('Encoder'):
                _x = conv_block(x, filters=16, sampling='same', **self.conv_block_params)
                _x = conv_block(_x, filters=16, sampling='down', **self.conv_block_params)

                _x = conv_block(_x, filters=32, sampling='same', **self.conv_block_params)
                _x = conv_block(_x, filters=32, sampling='down', **self.conv_block_params)

                current_shape = _x.get_shape().as_list()[1:]
                _x = flatten(_x)
                _x = dense(_x, 512, activation_='lrelu')
                encoded = dense(_x, self.latent_dim)

            with tf.variable_scope('Decoder'):
                _x = dense(encoded, 512, activation_='lrelu')
                _x = dense(_x, current_shape[0]*current_shape[1]*current_shape[2], activation_='lrelu')
                _x = reshape(_x, current_shape)

                _x = conv_block(_x, filters=32,
                                sampling=self.upsampling, **self.conv_block_params)
                _x = conv_block(_x, filters=16,
                                sampling='same', **self.conv_block_params)

                _x = conv_block(_x, filters=16,
                                sampling=self.upsampling, **self.conv_block_params)
                _x = conv_block(_x, filters=self.channel,
                                sampling='same', **self.last_conv_block_params)

            return encoded, _x


class FCAutoEncoder(AE):
    def __init__(self, input_shape,
                 latent_dim,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 is_dropout=False,
                 is_training=True):
        super().__init__(input_shape,
                         latent_dim,
                         last_activation,
                         normalization,
                         upsampling,
                         is_dropout,
                         is_training)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            with tf.variable_scope('Encoder'):
                _x = dense(x, 256, activation_='lrelu')
                _x = dense(_x, 64, activation_='lrelu')
                encoded = dense(_x, self.latent_dim)

            with tf.variable_scope('Decoder'):
                _x = dense(encoded, 64, activation_='lrelu')
                _x = dense(_x, 256, activation_='lrelu')
                _x = dense(_x, 784)

            return encoded, _x
