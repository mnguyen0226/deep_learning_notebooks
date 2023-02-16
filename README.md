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
  - Preprocessing: sequences of words can be encoded as binary vectors or other types of encoding
  - Stacks Dense layers with relu activation can solve a lot of problems.
  - In binary classification, model should end with a Dense layer with one unit and a `sigmoid` activation: the output of your model should be a scalar between 0 and 1, encoding a probability.
  - With a scalar sigmoid output on a binary classification, the loss function should use a `binary_crossentropy`.
  - `RMSprop` usually a good choice.
  - Overfit can happen to monitor the performance on the training and validation dataset via History.
- Multiclass Classification: Reuter Dataset
  - Categorical crossentropy is almost always the loss function you should use for such problems. It minimizes the distance between the probability distributions output by the model and the true distribution of the targets.
  - There are two ways to handle labels in multiclass classification:
    - Encoding the labels via categorical encoding (also known as one-hot encoding) and using `categorical_crossentropy` as a loss function
    - Encoding the labels as integers and using the `sparse_categorical_crossentropy` loss function
  - If you need to classify data into a large number of categories, you should avoid creating information bottlenecks in your model due to intermediate layers that 
are too small.
- Regression: Boston House Dataset
  - Regression is done using different loss functions than we used for classification.
  - `Mean squared error (MSE)` is a loss function commonly used for regression.
  - Similarly, evaluation metrics to be used for regression differ from those used for classification; naturally, the concept of accuracy doesn’t apply for regression. A common regression metric is `mean absolute error (MAE)`.
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
- Generalization: The Goal of ML
  - **Optimization**: to get the best performance on training dataset.
  - **Generalization**: to get the best performance on testing dataset.
  - Goal is to get good at generalization, but you have access to optimization only.
  - **Underfit**: the lower the loss on training data, the lower the loss on test data.
  - **Overfit**: the lower the loss on training data, the higher the loss on the test data. Happens when the data is noisy
    - Model trained on rare features can cause overfit.
    - Noisy features can cause overfit. You should do feature selection by measuring how informative the feature is with respect to the task (feature vs label), then keep the one that above the threshold.
    - Sparse data
  - ML model acts as python dict(), what about generalization? It's not about the model but about the structure of the info in real world: the info is highly structured. The manifolds hypothesis states that all natural data lies on a low-dimensional manifold within a high-dim space where it is encoded. This is why deep learning model can generalize.
  - If you work with data points that can be interpolated, you can start making sense of points you’ve never seen before by relating them to other points that lie close on the manifold. In other words, you can make sense of the totality of the space using only a sample of the space. You can use interpolation to fill in the blanks.
  - Why Deep Learning works? DL model is a tool for uncrumpling paper ball, aka disentangling latent manifold.
  - While deep learning is indeed well suited to manifold learning, the power to generalize is more a consequence of the natural structure of your data than a consequence of any property of your model. You’ll only be able to generalize if your data forms a manifold where points can be interpolated. The more informative and the less noisy your features are, the better you will be able to generalize, since your input space will be simpler and better structured. Data curation and feature engineering are essential to generalization.
  - Dataset with densely samplings: the model learned to approximates the latent space well, and interpolation leads to genelization. While sparse sampling makes the model does not match the latent space and leads to wrong interpolation. Interpolation == guess the next data point based on the existent data point
  - **Regularization**: When getting more data is not possible, it's best to add constaints on the moothness of hte model curve. If a network can only afford to memorize a small number of patterns or very regular patterns, the optimization process will force it to focus on the most prominent patterns, which better the chance of over it

- **Evaluating ML Model**
  - Way 1: Training, validation, test sets. Validation set is used to tune hyperparameters (number and size of layers). Note that if you tune too much, then the validation set will be "leaked" to the model, although you did not train on it. The test set must not be reviewed
  - Way 2: Simple holdout validation. Train 80% and evaluate on 20%.
     - This way does not perform well on small dataset. Why? Because if different random shuffling rounds of data before splitting end up yielding very different measurement of model's performance, then, you will have issue.
  - Way 3: K-fold validation. 
     - Methods is helpful when the model's performance is variance 
  - Way 4: Iterated K-fold validation with shuffling. You use when you have relatively little data available and you need to evalute your model as precisely as possible. How? To apply K-fold validation multiple times, shuffling data every time before splitting it K ways. You then evaluate P * K models 
     - Very expensive
  
- Improving Model Fit
  - Problem 1: Training doesn’t get started: your training loss doesn’t go down over time. 
    - Tune the gradient descent process: 
      - your choice of optimizer
      - the distribution of initial values in the weights of your model
      - your learning rate
      - your batch size
     
  - Problem 2: Training gets started just fine, but your model doesn’t meaningfully generalize: you can’t beat the common-sense baseline you set.
    -  something is fundamentally wrong with your approach
      - it may be that the input data you’re using simply doesn’t contain sufficient information to predict your targets: the problem as formulated is not solvable.
      - kind of model you’re using is not suited for the problem at hand
   
  - Problem 3: Training and validation loss both go down over time, and you can beat your baseline, but you don’t seem to be able to overfit, which indicates you’re still underfitting.
     - you can’t seem to be able to overfit, it’s likely a problem with the representational power of your model: you’re going to need a bigger model, one with more capacity, that is to say, one able to store more information
 
- Improve Generalization
  - Dataset Curation:
    - Get more data.
    - Minimize labeling error.
    - Deal with missing data.
    - Feature selection
  - Feature engineer: the process of using your own knowledge about the data and about the machine leanring algorithm at hand to make the algorithm owrk beter by applying hardcoded (non-learned) transformations to the data before it goes into the model. It's a basic way to make the proble easie by expressing it in a simpler way. Although deep learning models are able to extract features, you still need feature engineering.
    - Good features still allow you to solve problems more elegantly while using fewer resources.
    - Good features let you solve a problem with far less data.
  - Early Stopping: which will interrupt training as soon as validation metrics have stopped improving, while remembering the best known model state.
  - Regularization: are a set of best practices that actively impede the model’s ability to fit perfectly to the training data, with the goal of making the model perform better during validation. This is called “regularizing” the model, because it tends to make the model simpler, more “regular,” its curve smoother, more “generic”; thus it is less specific to the training set and better able to generalize by more closely approximating the latent manifold of the data.
    - Reducing network's size: If the model has limited memorization resources, it won’t be able to simply memorize its training data; thus, in order to minimize its loss, it will have to resort to learning compressed representations that have predictive power regarding the targets—precisely the type of representations we’re interested in.
      - You’ll know your model is too large if it starts overfitting right away and if its validation loss curve looks choppy with highvariance
      - Note: The bigger model starts overfitting almost immediately, after just one epoch, and it overfits much more severely. Its validation loss is also noisier. It gets training loss near zero very quickly. The more capacity the model has, the more quickly it can model the training data (resulting in a low training loss), but the more susceptible it is to overfitting (resulting in a large difference between the training and validation loss). 
    - Adding weight regularization:  Simpler models are less likely to overfit than complex ones. 
      - Simpler model = the model with fewer parameter.
      - A common way to mitigate overfit is to put constraints on complexity of the model by forcing its weight to take only small values, which makes the distribution of the weight value more regular. Weight regularization == adding the loss function of the model a cost associated with having large weights.
      - L1 regularization: The cost added is proportional to the absolute value of the weight coefficients
      - L2 regularization: The cost added is proportional to the square of the value of the weight coefficients L2 regularization is also called weight decay in the context of neural networks.
   - Dropout:

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
