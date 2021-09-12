import tensorflow as tf

input_shape = (1, 28, 28, 3)
x = tf.random.normal(input_shape)

from tensorflow.keras.layers import Layer


class BasicConv(Layer):
	def __init__(self, out_planes, kernel_size, stride=1, padding="same", dilation=1, groups=1, relu=True,
	             bn=True, bias=False):
		super(BasicConv, self).__init__()

		self.out_channels = out_planes
		self.conv = tf.keras.layers.Conv2D(filters=out_planes, kernel_size=kernel_size, strides=stride, padding=padding,
		                                   dilation_rate=dilation, groups=groups, use_bias=bias)
		self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if bn else None
		self.relu = tf.keras.layers.ReLU() if relu else None

	def __call__(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class ChannelPool(tf.Module(name=None)):
	def __call__(self, x):
		return tf.concat((tf.squeeze(tf.math.reduce_max(x, 1)[0]), tf.squeeze(tf.math.reduce_mean(x, 1))), axis=1)


class SpatialGate(tf.Module(name=None)):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 7
		self.compress = ChannelPool()
		self.spatial = BasicConv(out_planes=1, kernel_size=kernel_size, stride=1, relu=False)

	def __call__(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = tf.math.sigmoid(x_out)
		return x * scale


class TripletAttention(tf.keras.layers.Layer()):
	def __init__(self, no_spatial=False):
		super(TripletAttention, self).__init__()
		self.ChannelGateH = SpatialGate()
		self.ChannelGateW = SpatialGate()
		self.no_spatial = no_spatial
		self.Permute = tf.keras.layers.Permute()
		if not no_spatial:
			self.SpatialGate = SpatialGate()

	def call(self, x):
		x_perm1 = tf.transpose(x, perm=(0, 2, 1, 3))
		x_out1 = self.ChannelGateH(x_perm1)
		x_out11 = tf.transpose(x_out1, (0, 2, 1, 3))
		x_perm2 = tf.transpose(x, perm=(0, 3, 2, 1))
		x_out2 = self.ChannelGateW(x_perm2)
		x_out21 = tf.transpose(x_out2, (0, 3, 2, 1))
		if not self.no_spatial:
			x_out = self.SpatialGate(x)
			x_out = (1 / 3) * (x_out + x_out11 + x_out21)
		else:
			x_out = (1 / 2) * (x_out11 + x_out21)
		return x_out
