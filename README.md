# Kaggle Competitions

## Deep Learning with Python

### Chapter 2: The mathematical building blocks of neural networks
- Tensors from the foundation of the modern ML systems. They come in various flavors of dtype, rank, and shape.
- You can manipulate numerical tensors via tensor operations which can be interpreted as encoding geometric transformantions. Everything in DL is amenable to a geometric interpretation.
- DL models consist of chains of simple tensor operations, parameterized by weights, which are themselves tensors. The weights of a model are where its "knowledge" is stored.
- Learning means finding a set of values for a model's weights that minimize the loss function given a set of training samples and their corresponding target.
- Learning happens by drawing random batches of data samples and their targets, and computing the gradient of the model parameters with respect to the loss on the batch. The model parameters are then moved a bit in the opposite direction from gradient. This is called "mini-batch SGD".
- The entire learning process ois made possible by the fact that all tensor operations in NN are differentiable, thus it is possible to apply the chain rule of derivativation to find the gradient function mapping the current parameters and current batch of data to a gradient value.
- Loss is the quantity you'll attempt to minimize during training. It represents the measurement of success for the task you're trying to solve.
- Optimizer specifies the exact way in which the gradient of the loss with be used to update parameters: RMSProp or SGD or SGD with momenntum.

### Chapter 3: Introduction to Keras and Tensorflow
- TensorFlow is an industry-strength numerical computing framework that can run on CPU, GPU, or TPU. It can automatically compute the gradient of any differentiable expression, it can be distributed to many devices, and it can export programs to various external runtimes—even JavaScript.
- Key TensorFlow objects include tensors, variables, tensor operations, and the gradient tape.
- The central class of Keras is the Layer. A layer encapsulates some weights and some computation. Layers are assembled into models.
- Before you start training a model, you need to pick an optimizer, a loss, and some metrics, which you specify via the model.compile() method.
- To train a model, you can use the fit() method, which runs mini-batch gradient descent for you. You can also use it to monitor your loss and metrics on validation data, a set of inputs that the model doesn’t see during training.
- Once your model is trained, you use the model.predict() method to generate predictions on new inputs.

### Chapter 4: Getting Started With Neural Networks: Classification and Regression
- Binary Classification: IMDB Dataset
  - Preprocessing: sequences of words can be encoded as binary vectors or other types of enciding
  - Stacks Dense layers with relu activation can solve a lot of problems.
  - In binary classification, model should end with a Dense layer with one unit and a sigmoid activation: the output of your model should be a scalar between 0 and 1, encoding a probability.
  - With a scalar sigmoid output on a binary classification, the loss function should use a binary_crossentropy.
  - RMSprop usually a good choice.
  - Overfit can happen to monitor the performance on the training and validation dataset via History.
- Multiclass Classification: Reuter Dataset
  - Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.
  - There are two ways to handle labels in multiclass classification:
    - Encoding the labels via categorical encoding (also known as one-hot encoding) and using categorical_crossentropy as a loss function
    - Encoding the labels as integers and using the sparse_categorical_crossentropy loss function
  - If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that 
are too small.
- Regression: Boston House Dataset
  - Regression is done using different loss functions than we used for classification.
  - Mean squared error (MSE) is a loss function commonly used for regression.
  - Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is mean absolute error (MAE).
  - Normalization: When features in the input data have values in different ranges, each feature should be scaled independently as a preprocessing step
  - K-fold: When there is little data available, using K-fold validation is a great way to reliably evaluate a model. It can also be helpful to determine the right number of epochs to train prior to overfit.
  - When little training data is available, it’s preferable to use a small model with few intermediate layers (typically only one or two), in order to avoid severe overfitting. 
- Summary:
  - The three most common kinds of machine learning tasks on **vector data** are binary classification, multiclass classification, and scalar regression
  - Regression uses different loss functions and different evaluation metrics than classification.
  - You’ll usually need to preprocess raw data before feeding it into a neural network.
  - When your data has features with different ranges, scale each feature independently as part of preprocessing.
  - As training progresses, neural networks eventually begin to overfit and obtainworse results on never-before-seen data.
  - If you don’t have much training data, use a small model with only one or two intermediate layers, to avoid severe overfitting
  - If your data is divided into many categories, you may cause information bottlenecks if you make the intermediate layers too small.
  - When you’re working with little data, K-fold validation can help reliably evaluate your model.
  
### Chapter 5: Fundamentals of Machine Learning

### Chapter 6: The Universal Workflow of Machine Learning

### Chapter 7: Working With Keras: A Deep Dive

### Chapter 8: Introduction To Deep Learing For Computer Vision

### Chapter 9: Advanced Deep Learning For Computer Vision

### Chapter 10: Deep Learning for Time Series

### Chapter 11: Deep Learning for Text

### Chapter 12: Generative Deep Learning

### Chapter 13: Best Practices for Real World

### Chapter 14: Conclusion


## Participated Competitions
