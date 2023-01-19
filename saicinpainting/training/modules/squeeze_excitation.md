# SENet (Squeeze-and-Excitation Network)
- Source: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
- Squeeze-and-Excitation Networks (SENets) introduce a building block for CNNs that improves channel interdependencies at almost no computational cost. ***Besides this huge performance boost, they can be easily added to existing architectures.*** The main idea is this: ***Let’s add parameters to each channel of a convolutional block so that the network can adaptively adjust the weighting of each feature map.***
- CNNs use their convolutional filters to extract hierarchal information from images. *Lower layers find trivial pieces of context like edges or high frequencies, while upper layers can detect faces, text or other complex geometrical shapes.* They extract whatever is necessary to solve a task efficiently.
- ***All you need to understand for now is that the network weights each of its channels equally when creating the output feature maps. SENets are all about changing this by adding a content aware mechanism to weight each channel adaptively. In it’s most basic form this could mean adding a single parameter to each channel and giving it a linear scalar how relevant each one is.***
- *First, they get a global understanding of each channel by squeezing the feature maps to a single numeric value. This results in a vector of size n, where n is equal to the number of convolutional channels. Afterwards, it is fed through a two-layer neural network, which outputs a vector of the same size. These n values can now be used as weights on the original features maps, scaling each channel based on its importance.*
- ![SENet Architecture](https://miro.medium.com/max/658/1*WNk-atKDUsZPvMddvYL01g.png)
- Implementation
	```python
	def se_block(x, c, r=16):
		z = GlobalAveragePooling2D()(x)
        z = Dense(units=c//r, activation="relu")(z)
        z = Dense(units=c, activation="sigmoid")(z)
        return z*x
	```