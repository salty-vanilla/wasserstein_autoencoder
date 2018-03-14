import tensorflow as tf
import os
import csv
import time
import numpy as np
from PIL import Image


class WAE:
    def __init__(self,
                 autoencoder,
                 is_training=True):
        tf.reset_default_graph()
        self.autoencoder = autoencoder
        self.image_shape = self.autoencoder.input_shape
        self.latent_dim = self.autoencoder.latent_dim
        self.x = tf.placeholder(tf.float32, [None] + list(self.image_shape), name='x')
        self.z = tf.placeholder(tf.float32, [None, self.latent_dim], name='z')

        self.z_, self.y = self.autoencoder(self.x)

        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.model_dir = None
        self.is_training = is_training

    def fit_generator(self, *args, **kwargs):
        raise NotImplementedError

    def restore(self, file_path):
        tf.train.Saver().restore(self.sess, file_path)

    def predict(self, x, batch_size=16):
        outputs = np.empty([0] + list(self.image_shape))
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_on_batch(self, x):
        return self.sess.run(self.y,
                             feed_dict={self.x: x})

    def predict_latent_vectors(self, x, batch_size=16):
        outputs = np.empty([0, self.latent_dim])
        steps_per_epoch = len(x) // batch_size if len(x) % batch_size == 0 \
            else len(x) // batch_size + 1
        for iter_ in range(steps_per_epoch):
            x_batch = x[iter_ * batch_size: (iter_ + 1) * batch_size]
            o = self.predict_latent_vectors_on_batch(x_batch)
            outputs = np.append(outputs, o, axis=0)
        return outputs

    def predict_latent_vectors_on_batch(self, x):
        return self.sess.run(self.z_,
                             feed_dict={self.x: x})

    def visualize(self, dst_dir, image_batch, convert_function):
        decoded_data = self.predict_on_batch(image_batch)
        decoded_images = convert_function(decoded_data)
        original_images = convert_function(image_batch)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for i, (input_, output) in enumerate(zip(original_images, decoded_images)):
            dst_path = os.path.join(dst_dir, "{}.png".format(i))
            if len(input_.shape) == 2 or \
                    (len(input_.shape) == 3 and input_.shape[-1] == 3):
                image = np.concatenate((input_, output), axis=1)
            else:
                input_ = input_.transpose(0, 2, 1)
                output = output.transpose(0, 2, 1)
                input_ = input_.reshape(input_.shape[0], input_.shape[1]*input_.shape[2])
                output = output.reshape(output.shape[0], output.shape[1]*output.shape[2])
                image = np.concatenate((input_, output), axis=0)
            pil_image = Image.fromarray(np.uint8(image))
            pil_image.save(dst_path)

    def save(self, epoch):
        dst_dir = os.path.join(self.model_dir, "epoch_{}".format(epoch))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        return self.saver.save(self.sess, save_path=os.path.join(dst_dir, 'model.ckpt'))


class WAEGAN(WAE):
    def __init__(self,
                 autoencoder,
                 discriminator,
                 lambda_=0.1,
                 is_training=True,
                 is_summary=True):
        super().__init__(autoencoder, is_training)

        self.discriminator = discriminator
        self.discriminate_real = self.discriminator(self.z, reuse=False)
        self.discriminate_fake = self.discriminator(self.z_, reuse=True)

        with tf.name_scope('Loss'):
            self.loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminate_real,
                                                                       labels=tf.ones_like(self.discriminate_real))
            self.loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminate_fake,
                                                                       labels=tf.zeros_like(self.discriminate_fake))
            self.loss_d = (tf.reduce_mean(self.loss_d_real) + tf.reduce_mean(self.loss_d_fake)) / 2

            self.loss_g = tf.reduce_mean(tf.square(self.x - self.y), name='MSE')
            self.loss_q = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.discriminate_fake,
                                                        labels=tf.ones_like(self.discriminate_fake)))
            self.loss_ae = self.loss_g + lambda_*self.loss_q

        if is_training:
            with tf.name_scope('Optimizer'):
                self.opt_ae = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).\
                    minimize(self.loss_ae, var_list=self.autoencoder.vars)

                self.opt_d = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.5). \
                    minimize(self.loss_d, var_list=self.discriminator.vars)

        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if is_summary:
            tf.summary.FileWriter('../logs', graph=self.sess.graph)
        self.model_dir = None
        self.is_training = is_training

    def fit_generator(self, image_sampler,
                      latent_sampler,
                      nb_epoch=1000,
                      save_steps=5,
                      visualize_steps=5,
                      result_dir='result',
                      model_dir='model'):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.n
        self.model_dir = model_dir

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        f = open(os.path.join(result_dir, 'learning_log.csv'), 'w')
        writer = csv.writer(f, lineterminator='\n')

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1
        writer.writerow(['loss_d', 'loss_ae', 'loss_g', 'loss_q'])

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                image_batch = image_sampler()
                latent_batch = latent_sampler(image_batch.shape[0], self.latent_dim)

                _, loss_d = self.sess.run([self.opt_d, self.loss_d],
                                          feed_dict={self.x: image_batch,
                                                     self.z: latent_batch})

                _, loss_ae, loss_g, loss_q = self.sess.run([self.opt_ae, self.loss_ae, self.loss_g, self.loss_q],
                                                           feed_dict={self.x: image_batch})

                print('iter : {} / {}  {:.1f}[s]  '
                      'loss_d : {:.4f}  loss_ae : {:.4f}  loss_g : {:.4f}  loss_q : {:.4f}\r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_d, loss_ae, loss_g, loss_q),
                      end='')
                writer.writerow([loss_d, loss_ae, loss_g, loss_q])
                f.flush()

            if visualize_steps is not None:
                if epoch % visualize_steps == 0:
                    self.visualize(os.path.join(result_dir, 'epoch_{}'.format(epoch)),
                                   image_batch, image_sampler.data_to_image)

            if epoch % save_steps == 0:
                self.save(epoch)

        print('\nTraining is done ...\n')


class MMDWAE(WAE):
    def __init__(self,
                 autoencoder,
                 kernel,
                 lambda_=0.1,
                 is_training=True,
                 is_summary=True):
        super().__init__(autoencoder, is_training)

        with tf.name_scope('Loss'):
            self.loss_g = tf.reduce_mean(tf.square(self.x - self.y), name='MSE')

            with tf.name_scope('MMD'):
                self.n = tf.placeholder(tf.float32, [])
                loss_x_x = kernel(tf.expand_dims(self.z, axis=1),
                                  tf.expand_dims(self.z, axis=-1))
                loss_x_x *= 1 - tf.eye(self.latent_dim)
                loss_x_x = tf.reduce_sum(loss_x_x) / (self.n*(self.n-1))

                loss_y_y = kernel(tf.expand_dims(self.z_, axis=1),
                                  tf.expand_dims(self.z_, axis=-1))
                loss_y_y *= 1 - tf.eye(self.latent_dim)
                loss_y_y = tf.reduce_sum(loss_y_y) / (self.n*(self.n-1))

                loss_x_y = kernel(tf.expand_dims(self.z, axis=1),
                                  tf.expand_dims(self.z_, axis=-1))
                loss_x_y = tf.reduce_sum(loss_x_y) / (self.n*self.n)

                self.loss_mmd = loss_x_x + loss_y_y - 2 * loss_x_y

            self.loss_ae = self.loss_g + lambda_*self.loss_mmd

        if is_training:
            with tf.name_scope('Optimizer'):
                self.opt_ae = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).\
                    minimize(self.loss_ae, var_list=self.autoencoder.vars)

        self.saver = tf.train.Saver(max_to_keep=None)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if is_summary:
            tf.summary.FileWriter('../logs', graph=self.sess.graph)
        self.model_dir = None
        self.is_training = is_training

    def fit_generator(self, image_sampler,
                      latent_sampler,
                      nb_epoch=1000,
                      save_steps=5,
                      visualize_steps=5,
                      result_dir='result',
                      model_dir='model'):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.n
        self.model_dir = model_dir

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        f = open(os.path.join(result_dir, 'learning_log.csv'), 'w')
        writer = csv.writer(f, lineterminator='\n')

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1
        writer.writerow(['loss_ae', 'loss_g', 'loss_mmd'])

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                image_batch = image_sampler()
                latent_batch = latent_sampler(image_batch.shape[0], self.latent_dim)

                _, loss_ae, loss_g, loss_mmd =\
                    self.sess.run([self.opt_ae, self.loss_ae, self.loss_g, self.loss_mmd],
                                  feed_dict={self.x: image_batch,
                                             self.z: latent_batch,
                                             self.n: image_batch.shape[0]})

                print('iter : {} / {}  {:.1f}[s]  '
                      'loss_ae : {:.4f}  loss_g : {:.4f}  loss_mmd : {:.4f}\r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_ae, loss_g, loss_mmd),
                      end='')
                writer.writerow([loss_ae, loss_g, loss_mmd])
                f.flush()

            if visualize_steps is not None:
                if epoch % visualize_steps == 0:
                    self.visualize(os.path.join(result_dir, 'epoch_{}'.format(epoch)),
                                   image_batch, image_sampler.data_to_image)

            if epoch % save_steps == 0:
                self.save(epoch)

        print('\nTraining is done ...\n')