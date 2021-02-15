# How much impact Batch normalization on Training?

I tested  
1. how much improve MNIST classification accuracy.
2. How long increase training time on same network model except batchnormalization layers


## Network model

**Convolution layers** : stride = 1, kernel_size = 3, activation function = relu
 
**MaxPool** : stride = 2, kernel_size = 2

### Basic Convolutional Neural Network (CNN ver-1)

**First block** : 3*3,conv2d,32 --> 3*3,conv2d,32 --> maxpool,/2

**Second block**: 3*3,conv2d,64 --> 3*3,conv2d,64 --> maxpool,/2

**Third block** : 3*3,conv2d,128 --> 3*3,conv2d,128 --> 3*3,conv2d,128 --> 3*3,conv2d,128 --> maxpool,/2 

**Fully connected Block** : fc 512 --> fc 128 --> softmax 10 


### CNN with batchnormalization (CNN ver-2)

**First block** : 3*3,conv2d,32 --> bn --> 3*3,conv2d,32 --> maxpool,/2

**Second block**: 3*3,conv2d,64 --> bn --> 3*3,conv2d,64 --> maxpool,/2

**Third block** : 3*3,conv2d,128 --> bn --> 3*3,conv2d,128 --> 3*3,conv2d,128 --> 3*3,conv2d,128 --> maxpool,/2 

**Fully connected Block** : fc 512 --> fc 128 --> softmax 10 

### CNN with Batchnormalization between all layers (CNN ver-3)

**First block** : 3*3,conv2d,32 --> bn --> 3*3,conv2d,32 --> bn --> maxpool,/2

**Second block**: 3*3,conv2d,64 --> bn --> 3*3,conv2d,64 --> bn --> maxpool,/2

**Third block** : 3*3,conv2d,128 --> bn --> 3*3,conv2d,128 --> bn --> 3*3,conv2d,128 --> bn --> 3*3,conv2d,128 --> bn --> maxpool,/2 

**Fully connected Block** : fc 512 --> bn --> fc 128 --> bn --> softmax 10 



## Experiment

### Hardware 
CPU : i5-8400

GPU : nvidia Geforce GTX 1060 3G

### Dataset : MNIST
**No augmentation**


### Hyperparameter
batch_size = 256

learning rate = 0.0001

n_epochs = 20

### criterion & optimizer

**loss func** : CrossEntropyLoss

**optimizer** : ADAM

## Result

|   | CNN ver-1  | CNN ver-2  |  CNN ver-3  |
|---|---|---|---|
| Accuracy(%)  |  98.33 | 98.72  | 99.2  |
| Training time(s)  | 242  | 256  | 273  |









