import tensorflow as tf
import matplotlib.pyplot as plt

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

image = tf.keras.preprocessing.image.load_img(
	"F:\Pycharm_projects\Gnet2\data\WhatsApp Image 2021-09-12 at 1.01.31 PM (1).jpeg")
plt.imshow(image)
plt.show()
x = tf.expand_dims(tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(image)), axis=0)
print(x.shape)


class BasicConv(tf.keras.Model):
	def __init__(self, out_planes, kernel_size, stride=1, padding="same", dilation=1, groups=1, relu=True,
	             bn=True, bias=False):
		super(BasicConv, self).__init__()

		self.out_channels = out_planes
		self.conv = tf.keras.layers.Conv2D(filters=out_planes, kernel_size=kernel_size, strides=stride, padding=padding,
		                                   dilation_rate=dilation, groups=groups, use_bias=bias)
		self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if bn else None
		self.relu = tf.keras.layers.ReLU() if relu else None

	def call(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


import tensorflow as tf
import torch


class ChannelPool(tf.keras.Model):
	def __init__(self):
		super(ChannelPool, self).__init__()

	def __call__(self, x):
		x = x.numpy()
		x = torch.from_numpy(x)
		x = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
		x = x.numpy()
		x = tf.convert_to_tensor(x)

		return x


class SpatialGate(tf.keras.Model):
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


class TripletAttention(tf.keras.Model):
	def __init__(self, no_spatial=False):
		super(TripletAttention, self).__init__()
		self.ChannelGateH = SpatialGate()
		self.ChannelGateW = SpatialGate()
		self.no_spatial = no_spatial
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


model = TripletAttention()
x = model(x=x)
plt.imshow(tf.squeeze(x).numpy() / 255)
plt.show()
