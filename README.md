# AutoEncoderPractice
A practice session to learn autoencoder and variational autoencoder.

# Denoising MNIST:
Noise Reduction of MNIST dataset using convolutional autoencoder.

![Sample Noise Reduction Output](output_sample.png)
The first row indicates the MNIST digits with some added noise. The second row is the reconstructed noise free image using autoencoder.

## Architecture:

Input --> conv1 --> pool1 --> conv2 --> pool2 --> conv3 --> pool3 -->conv4 --> upsampling_1 --> conv5 --> upsampling_2 --> conv6 --> upsampling_3 --> Output

Input => (32, 32, 1) MNIST data with gaussian noise  
conv1 => Convolutional Layer with shape (32, 3, 3)  
pool1 => Maxpooling Layer with size (2, 2)  
conv2 => Convolutional Layer with shape (16, 3, 3)  
pool2 => Maxpooling Layer with size (2, 2)  
conv3 => Convolutional Layer with shape (16, 3, 3)  
pool3 => Maxpooling Layer with size (2, 2)  
conv4 => Convolutional Layer with shape (16, 3, 3)  
upsampling_1 => UpSampling Layer with size (2, 2)  
conv5 => Convolutional Layer with shape (16, 3, 3)  
upsampling_2 => UpSampling Layer with size (2, 2)  
conv6 => Convolutional Layer with shape (32, 3, 3)  
upsampling_3 => UpSampling Layer with size (2, 2)  
Output => Convolutional Layer with shape (1, 3, 3) MNIST data without noise  
