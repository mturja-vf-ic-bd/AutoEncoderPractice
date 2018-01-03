import tensorflow as tf
import numpy as np
from keras import Model
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, UpSampling2D, Input, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras import losses

conv_depth_1 = 32
conv_depth_2 = 16
k_size = 3
batch_size = 256
epoch = 30

(X_train, y_train), (_, _) = mnist.load_data() # fetch MNIST data

num_train, height, width = X_train.shape
depth = 1
X_train = X_train.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
x_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_train = np.expand_dims(X_train, 3)
X_train_noisy = np.expand_dims(X_train_noisy, 3)
print('Number of Training Examples', num_train)
print('Shape of Training Examples', X_train.shape)


inp = Input(shape=(width, height, depth))
conv_1 = Conv2D(conv_depth_1, (k_size, k_size), padding ='same', activation ='relu')(inp)
pool_1 = MaxPooling2D((2, 2), padding ='same')(conv_1)
conv_2 = Conv2D(conv_depth_2, (k_size, k_size), padding = 'same', activation = 'relu')(pool_1)
pool_2 = MaxPooling2D((2, 2), padding ='same')(conv_2)
conv_3 = Conv2D(conv_depth_2, (k_size, k_size), padding = 'same', activation = 'relu')(pool_2)
encoder = MaxPooling2D((2, 2), padding ='same')(conv_3)
conv_4 = Conv2D(conv_depth_2, (k_size, k_size), padding='same', activation='relu')(encoder)
upSample_1 = UpSampling2D(size = (2, 2))(conv_4)
conv_5 = Conv2D(conv_depth_2, (k_size, k_size), padding='same', activation='relu')(upSample_1)
upSample_2 = UpSampling2D(size = (2, 2))(conv_5)
conv_6 = Conv2D(conv_depth_1, (k_size, k_size), activation='relu')(upSample_2)
upSample_3 = UpSampling2D(size = (2, 2))(conv_6)
out = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(upSample_3)

model = Model(inputs = inp, outputs = out)
model.compile(optimizer='adam', loss=losses.binary_crossentropy)
model.fit(X_train_noisy, X_train, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.1)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


