import tensorflow as tf


def linear():
    def f(x, y):
        return tf.matmul(x, y)
    return f


def rbf(gamma=1):
    def f(x, y):
        return tf.exp(-gamma * tf.square(x - y))
    return f


def polynomial(r=1, d=2):
    def f(x, y):
        return tf.pow((tf.matmul(x, y) + r), d)
    return f
