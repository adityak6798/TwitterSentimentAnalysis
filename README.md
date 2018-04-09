# TwitterSentimentAnalysis
A deep learning model to implement Sentiment Analysis by a character-level Convolutional Neural Network, based on the paper by Xiang Zhang, Yann LeCun and Junbo Zhao.(https://arxiv.org/abs/1509.01626) Model is scaled down to decrease training time and memory usage.

This model has been implemented using Keras(Tensorflow backend), Pandas and Numpy. 
Model Architecture:
1) 1D Conv Layer with 64 kernels,  kernel_size=7, ReLU activation.
2) 1D Max Pooling with pool_size=3
3) 1D Conv Layer with 64 kernels,  kernel_size=7,ReLU activation.
4) 1D Max Pooling with pool_size=3
5) 1D Conv Layer with 32 kernels, kernel_size=3,ReLU activation 
6) 1D Conv Layer with 32 kernels, kernel_size=3,ReLU activation
7) 1D Conv Layer with 16 kernels, kernel_size=3,ReLU activation
8) 1D Conv Layer with 16 kernels, kernel_size=3,ReLU activation
9) 1D Max Pooling with pool_size=3
10) Flattening
11) Dense Layer with 64 nodes, ReLU activation, 0.5 dropout
13) Dense Layer with 32 nodes, ReLU activation, 0.5 dropout
14) Output Node with sigmoid activation
Binary Cross-entropy loss function, RMSProp optimizer. Trained for 10 epochs with 60000 examples split into 54000 training and 6000 cross-validation examples. Batch size 64.
Expected Accuracy: Around 75%
