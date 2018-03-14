import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.utils import to_categorical
from image_sampler import normalize, denormalize


class ImageSampler(Iterator):
    def __init__(self, batch_size=64,
                 shuffle=True,
                 is_training=True,
                 is_vectorize=False,
                 normalize_mode='tanh'):
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        if is_training:
            self.x = train_x
            self.y = train_y
        else:
            self.x = test_x
            self.y = test_y
        self.is_training = is_training
        self.normalize_mode = normalize_mode

        if is_vectorize:
            # x: (N, 28, 28) → (N, 784)
            self.x = self.x.reshape(self.x.shape[0], self.x.shape[1]*self.x.shape[2])
        else:
            self.x = np.expand_dims(self.x, axis=-1)
            # one-hot vectorize y
        self.y = to_categorical(self.y, 10)
        super().__init__(len(self.x), batch_size, shuffle, None)

    def __call__(self, *args, **kwargs):
        """
        if is_training: return (image_batch, label_batch)
        else          : return (generator yields  (image_batch, label_batch))
        """
        if self.is_training:
            return self._flow_on_training()
        else:
            return self._flow_on_test()

    def _flow_on_training(self):
        # get minibatch indices
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        image_batch = np.array(self.x[index_array])
        image_batch = normalize(image_batch, self.normalize_mode)
        # label_batch = np.array(self.y[index_array])
        return image_batch

    def _flow_on_test(self):
        # create indices
        indexes = np.arange(self.n)

        # calculate steps per a test
        steps = self.n // self.batch_size
        if self.n % self.batch_size != 0:
            steps += 1

        # yield loop
        for i in range(steps):
            index_array = indexes[i*self.batch_size: (i+1)*self.batch_size]
            image_batch = self.x[index_array]
            image_batch = normalize(image_batch, self.normalize_mode)
            # label_batch = self.y[index_array]
            yield image_batch

    def data_to_image(self, x):
        if x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])
        return denormalize(x, self.normalize_mode)
