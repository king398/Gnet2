import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass
# Load a model
image_size = 768
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
	image_size=image_size,
	activation='sigmoid',
	pretrained=True,
	include_top=True,
	pretrained_top=True
)
classes = utils.get_imagenet_classes()

# Get an image and compute the attention map
url = "https://www.kaggleusercontent.com/kf/73815717/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..0YpIm7-WK6hlH8Gh3HrRUw.-j12U71IXov94KnFG919gzQuagaW2ByLyJKS9uURdTvbHyEPxTsmq0xCnaXXfTQq2AJyTnIaI_hvlUp-GjDuSHwRG6YVH26JW4cvgOY1lQOgF3Zc6943GdhPYH5YWMLuV8IYVuriV0nyt8XOKziqXC6Hc95xPpWpYSD0XGYzObkR8SS_7WUWlwJ28NMbxudICYxovUNHZKjUv374vDHMDDO888DUqDRl6MHefS5jmQSCUUbHbi-2--SRhg4AMCefcNStYiJhMvLAjpCvjN9tS__M0wsm4-MJz1tv5naVLsHUhkUmep9ZhlkV-GzDNtXPlVgHNO4stOyRauawXfINrTl-LVqorooS_iApoZUN3Yr01j5zWqzxTXDzbSVG2F_6doNTHiO7SCnVkL5UBCP8G55rm9WIY4uBaHou8f0we7kousjylCLYcWvwgoDvRm4PPr3Y_JuUhiIzGYkm-JjCki8lPC8viB7s-JsLogDMqBaZAZTEUdQdB3Iyg7-u-nSAlGmbBPEgMoKW1fadO4rOzWCwxzgjCQSZI306YHK8uhox7RrQZPdJszaorJpsU7EzJnl9-KmOC_EryKxgzd14GJYpxfN2OYCtwZCPH9XaAgYpi81CYhkrswMc8eLMuymw74hkcJmHtUhBuRBvSnS5oZIvbPF7pkDV8TVivniXeNO1O6iToqp5SSGCqSCqE9dU.V951CAvPNy6w0Cfx2ADVng/__results___files/__results___16_0.png"
image = utils.read(url, image_size)
attention_map = visualize.attention_map(model=model, image=image)
print('Prediction:', classes[
	model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
      )  # Prediction: Eskimo dog, husky

# Plot results
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(image)
_ = ax2.imshow(attention_map)
plt.show()
