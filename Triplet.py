import tensorflow as tf
import matplotlib.pyplot as plt


class SelfAttention(tf.keras.layers.Layer):
	def __init__(self, n_channels):
		super(SelfAttention, self).__init__()
		self.query, self.key, self.value = [self._conv(n_channels, c) for c in
		                                    (n_channels // 8, n_channels // 8, n_channels)]

	def build(self, input_shape):
		self.kernel = self.add_weight(name='gamma ',
		                              s
