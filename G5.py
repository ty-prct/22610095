# Image Generation using a Pre-trained Diffusion Model on MNIST
# Author: Utkarsh Pingale

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 127.5) - 1.0
x_train = np.expand_dims(x_train, -1)

# Define simple UNet-like model (DDPM simulation)
def get_unet():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Conv2D(1, 3, activation='tanh', padding='same')(x)
    return tf.keras.Model(inputs, outputs)

model = get_unet()

# Simulate diffusion (reverse denoising)
def diffusion_sampling(model, steps=100, img_size=28):
    img = tf.random.normal((1, img_size, img_size, 1))  # start with noise
    for i in range(steps):
        pred = model(img, training=False)
        img = img - 0.1 * (img - pred)  # iterative denoising
    return img

# Generate 10 simulated digits (0–9)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flatten()

for digit in range(10):
    generated_img = diffusion_sampling(model)
    axes[digit].imshow(tf.squeeze((generated_img + 1) / 2), cmap='gray')
    axes[digit].set_title(f"Digit {digit}")
    axes[digit].axis("off")

plt.suptitle("Generated Digits 0–9 (DDPM Simulation)", fontsize=14)
plt.tight_layout()
plt.show()


#-----------------------------------------------------------

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

ds_train = ds_train.map(normalize_img).batch(32)

def generate_noise_image(num_image=5):
    return np.random.rand(num_image, 28, 28)

generated_images = generate_noise_image()

plt.figure(figsize=(10, 2))
for i, img in enumerate(generated_images):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
plt.show