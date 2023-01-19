# ResNet (Residual Neural Network)
- Source: https://en.wikipedia.org/wiki/Residual_neural_network, https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/#:~:text=Skip%20Connections%20(or%20Shortcut%20Connections,input%20to%20the%20next%20layers.&text=Neural%20networks%20can%20learn%20any,%2Ddimensional%20and%20non%2Dconvex, https://www.analyticsvidhya.com/blog/2021/06/understanding-resnet-and-analyzing-various-models-on-the-cifar-10-dataset/#h2_3
- Residual Networks were proposed in 2015 to solve the image classification problem. ****In ResNets, the information from the initial layers is passed to deeper layers by matrix addition. This operation doesn’t have any additional parameters as the output from the previous layer is added to the layer ahead.

## Skip Connection
- ***There are two main reasons to add skip connections: to avoid the problem of vanishing gradients, or to mitigate the Degradation (accuracy saturation) problem; where adding more layers to a suitably deep model leads to higher training error.***
- *Skipping effectively simplifies the network, using fewer layers in the initial training stages. This speeds learning by reducing the impact of vanishing gradients, as there are fewer layers to propagate through.*
- *While training deep neural nets, the performance of the model drops down with the increase in depth of the architecture. This is known as the degradation problem.*
- From this construction, *the deeper network should not produce any higher training error than its shallow counterpart because we are actually using the shallow model’s weight in the deeper network with added identity layers. But experiments prove that the deeper network produces high training error comparing to the shallow one. This states the inability of deeper layers to learn even identity mappings.*
- The degradation of training accuracy indicates that not all systems are similarly easy to optimize.
- *One of the primary reasons is due to random initialization of weights with a mean around zero, L1, and L2 regularization.  As a result, the weights in the model would always be around zero and thus the deeper layers can’t learn identity mappings as well.* Here comes the concept of skip connections which would enable us to train very deep neural networks.
- Skip Connections (or Shortcut Connections) as the name suggests skips some of the layers in the neural network and feeds the output of one layer as the input to the next layers.
- Skip Connections were introduced to solve different problems in different architectures. In the case of ResNets, skip connections solved the degradation problem that we addressed earlier.
- As you can see here, the loss surface of the neural network with skip connections is smoother and thus leading to faster convergence than the network without any skip connections.
- Skip Connections can be used in 2 fundamental ways in Neural Networks: Addition and Concatenation.
- One problem that may happen is regarding the dimensions. *Sometimes the dimensions of `x` and `F(x)` may vary and this needs to be solved.* Two approaches can be followed in such situations. *One involves padding the input x with weights such as it now brought equal to that of the value coming out. The second way includes using a convolutional layer from `x` to addition to `F(x)`*.

## Bottleneck Design
- Source: https://towardsdatascience.com/review-resnet-winner-of-ilsvrc-2015-image-classification-localization-detection-e39402bfa5d8
- Since the network is very deep now, the time complexity is high. A bottleneck design is used to reduce the complexity.
- The 1×1 conv layers are added to the start and end of network. This is a technique suggested in Network In Network and GoogLeNet (Inception-v1). ***It turns out that 1×1 conv can reduce the number of connections (parameters) while not degrading the performance of the network so much. (Please visit my review if interested.)***
- *With the bottleneck design, 34-layer ResNet become 50-layer ResNet. And there are deeper network with the bottleneck design: ResNet-101 and ResNet-152.*
- ![ResNet Architecture](https://neurohive.io/wp-content/uploads/2019/01/resnet-architectures-34-101.png)
- Implementation of 50-layered ResNet (for CIFAR-10)
	```python
	def residual_block(x, filters): 
		z1 = Conv2D(filters=filters[0], kernel_size=1, strides=1, padding="same")(x) 
		z1 = BatchNormalization()(z1)
		z1 = Activation("relu")(z1)
		z1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, padding="same")(z1)
		z1 = BatchNormalization()(z1)
		z1 = Activation("relu")(z1)
		z1 = Conv2D(filters=filters[1], kernel_size=1, strides=1, padding="same")(z1)
		z1 = BatchNormalization()(z1)
		z1 = Activation("relu")(z1)

		z2 = Conv2D(filters=filters[1], kernel_size=1, strides=1, padding="same")(x)
		z2 = BatchNormalization()(z2)

		z = Activation("relu")(z1 + z2)
		return z

	inputs = Input(shape=(32, 32, 3))

	z = Conv2D(filters=64, kernel_size=7, strides=2, padding="valid")(inputs)
	z = MaxPool2D(pool_size=3, strides=2, padding="same")(z)
	for _ in range(3):
		z = residual_block(z, filters=[64, 256])
	for _ in range(4):
		z = residual_block(z, filters=[128, 512])
	for _ in range(6):
		z = residual_block(z, filters=[256, 1024])
	for _ in range(3):
		z = residual_block(z, filters=[512, 2048])
	z = GlobalAveragePooling2D()(z) 

	outputs = Dense(units=10, activation="softmax")(z)

	model = Model(inputs=inputs, outputs=outputs)
	```