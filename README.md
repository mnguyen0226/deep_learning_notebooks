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
  
### Chapter 5: Fundamentals of Machine Learning (Important for ML Optimization Interview)
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
      
    ```python
    # original
    model.compile(optimizer=keras.optimizers.RMSprop(1.), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # better lr
    model.compile(optimizer=keras.optimizers.RMSprop(1e-2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    ```
     
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
      ```python
      # adding L2 weight regularization
      model = keras.Sequential([
       layers.Dense(16,
       kernel_regularizer=regularizers.l2(0.002),
       activation="relu"),
       layers.Dense(16,
       kernel_regularizer=regularizers.l2(0.002),
       activation="relu"),
       layers.Dense(1, activation="sigmoid")
      ])
      
      # L1 regulariztion
      regularizers.l1(0.001)
      
      # simultaneous L1 and L2 regularization
      regularizers.l1_l2(l1=0.001, l2=0.001)  
      ```
   - Dropout: One of the most effective and most used regularization techniques. Randomly dropout (set to 0) a number of ouptu features of the layer during training. At test time, no units are dropped out, instead the layer's output values are scaled down by a factor equal to the dropout rate, to balance for th fact that more units are active than at training time.
      ```python
      # at traiing drop 50% of unit
      layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
      
      # at testing scale 50%
      layer_output *= 0.5 
      
      model = keras.Sequential([
       layers.Dense(16, activation="relu"),
       layers.Dropout(0.5),
       layers.Dense(16, activation="relu"),
       layers.Dropout(0.5),
       layers.Dense(1, activation="sigmoid")
      ])
      ```
- Summary:
  - Common ways to maximize generalization and prevent overfit 
    - Get more training data or better training data.
    - Dev better feature.
    - Reduce the capacity of the model.
    - Add weight regularization (for smaller models).
    - Add dropout.
  - The purpose of a machine learning model is to generalize: to perform accurately on never-before-seen inputs. It’s harder than it seems.
  - A deep neural network achieves generalization by learning a parametric model that can successfully interpolate between training samples—such a model can be
said to have learned the “latent manifold” of the training data. This is why deep learning models can only make sense of inputs that are very close to what
they’ve seen during training.
  - The fundamental problem in machine learning is the tension between optimization and generalization: to attain generalization, you must first achieve a good fit to
the training data, but improving your model’s fit to the training data will inevitably start hurting generalization after a while. Every single deep learning best
practice deals with managing this tension.
  - The ability of deep learning models to generalize comes from the fact that they manage to learn to approximate the latent manifold of their data, and can thus
make sense of new inputs via interpolation.
  - It’s essential to be able to accurately evaluate the generalization power of your model while you’re developing it. You have at your disposal an array of evaluation methods, from simple holdout validation to K-fold cross-validation and iterated K-fold cross-validation with shuffling. Remember to always keep a completely separate test set for final model evaluation, since information leaks from your validation data to your model may have occurred.
  - When you start working on a model, your goal is first to achieve a model that has some generalization power and that can overfit. Best practices for doing
this include tuning your learning rate and batch size, leveraging better architecture priors, increasing model capacity, or simply training longer.
  - As your model starts overfitting, your goal switches to improving generalization through model regularization. You can reduce your model’s capacity, add dropout
or weight regularization, and use early stopping. And naturally, a larger or better dataset is always the number one way to help a model generalize.

### Chapter 6: The Universal Workflow of Machine Learning (Important for ML System Design)
- Universal workflow:
  - 1. Define the task: Understand the problem domain and the business logic underlying what the customer asked for. Collect a dataset, understand what the data represents, and choose how you will measure success on the task
  - 2. Develop a model: Prepare your data so that it can be processed by a ML model, select a model evaluation protocal and a simple baseline to beat. Train a first model that has generalization power and that can overfit, and then regularize and tune your model until you achieve the best possible generalization performance.
  - 3. Deploy the model: Present your work to stakeholders, ship the model to a webserver, a mobile app, a web page, or an embedded device, monitor the model's performance in the wild, and start collecting the data you will nedd to build the next generation model.
- The hardest things in machine learning are framing problems and collecting, annotating, and cleaning data.

- Define the task
  - In real-world, you won't have the dataset, you start from a prolem.
  - Frame the problem: Discuss with stakeholder
    - What will your input data be? What are you trying to predict? 
    - What type of machine learning task are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Image segmentation? Ranking? Something else, like clustering, generation, or reinforcement learning?
    - What do existing solutions look like?
    - Are there particular constraints you will need to deal with? 
  - Collect a Dataset (most arduous, time-consuming, and costly in ML project)
    - A good dataset is an asset worthy of care and investment. If you get an extra 50 hours to spend on a project, chances0 are that the most effective way to allocate them is to collect more data rather than search for incremental modeling improvements.
    - “The Unreasonable Effectiveness of Data” (2009 Google Research): data matters more than algorithms.
    - If possible, collect data directly from the environment where your model will be used.
    - Concept drift occurs when the properties of the production data change over time, causing model accuracy to gradually decay.  Dealing with fast concept drift requires constant data collection, annotation, and model retraining.
  - Understand your data
  - Choose a measure of success
    - Your metric for success will guide all of the technical choices you make throughout the project. It should directly align with your higher-level goals, such as the business success of your customer.
    - For a balanced classification, use ROC and accuracy
    - For imbalance classification, ranking or multilabel classification, use precision and recall, weighted accuracy, and ROC AUC.

- Develop a model
  - Vectorization: Convert the dataset into tensor.
  - Value Normalization: It isn't safe to feed into the network the data with large value as it can trigger large gradient update and prevent the model to converge. To helps the model learn:
    - Take small values—Typically, most values should be in the 0–1 range.
    - Be homogenous—All features should take values in roughly the same range.
    - Normalize each feature independently to have a mean of 0.
    - Normalize each feature independently to have a standard deviation of 1.
    ```python
    # x is a 2D matrix (samples, features)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    ```
  - Handling Missing Value:
    - You can just discard the feature OR
    - If the feature is categorical, create a new feature with "value missing"
    - If feature is numerical, don't put "0" as it may create a discontinuity in the latent space formed by your features, making it harder for a model trained on it to generalize. Replace it with the average or median value.
  - Choose an evaluation protocol: The goal of your validation protocol is to accurately estimate what your success metric of choice (such as accuracy) will be on actual production data. 
    - Maintaining a holdout validation set. This is the way to go when you have plenty of data.
    - Doing K-fold cross-validation. This is the right choice when you have too few samples for holdout validation to be reliable.
    - Doing iterated K-fold validation. This is for performing highly accurate model evaluation when little data is available. (less than the one using K-Fold)
  - Pick the right loss functions and activation function:
    - Binary classification:
      + Last-layer activation: sigmoid
      + Loss Function: binary_crossentropy
    - Multiclass, single-label classification:
      + Last-layer activation: softmax
      + Loss Function: categorical_crossentropy
    - Multiclass, multilabel classification:
      + Last-layer activation: sigmoid
      + Loss Function: binary_crossentropy
  - Develop a model that overfit
    - Add layers.
    - Make the layers bigger.
    - Train for more epochs.
    - Monitor the training loss and validation loss and the validation metrics. When you see that the model's performance on the validation data begin to degrade, you've achieved overfitting.
  - Regularize and tune your model
    - Once your model is overfir, you need to maximize generalization performance. Keep modify model -> train -> evaluate till you can't improve anymore
    - Try:
      - Different architectures; add or remove layers.
      - Add dropout
      - If the model is small, add L1 or L2 regularization
      - Try different hyperparameters (number of units per layer, learning rate,...)
      - Feature engineer: dev better feature, feature selection
      - Don't evaluate and tune your model too much
  - After all, train model on all dataset and evaluate on the test set. If test set evaluation is bad, this means that the validation set is not reliable or you overfit the validation set. In this case, you will need to switch to a better evaluation protocol (iterated K-fold validation)

- Deploy a the model
  - Explain your work to stakeholders and set expectations
    - Show example of failure mode.
    - Show metrics on true negative and false negative
    - Relate the model's performance metrics to business goals.
  - Ship an inference model:
    - Transfer code from colab to Python, mobile, embedded system.
    - Deploy a model as REST API
  - Deploying a model on a device
  - Deploying a model in the Browswer:
  - Maintain your model: '
    - Avoid concept drift: over time, the characteristics of your production data will change, gradually degrading the performance and relevance of your model.
    - Keep collecting and annotating data, and keep improving your annotation pipeline over time.
    - Watch out for changes in the production data. Are new features becoming available?

- Summary
  - When you take on a new machine learning project, first define the problem at hand:
    - Understand the broader context of what you’re setting out to do—what’s the end goal and what are the constraints?
    - Collect and annotate a dataset; make sure you understand your data in depth.
    -  Choose how you’ll measure success for your problem—what metrics will you monitor on your validation data?
  - Once you understand the problem and you have an appropriate dataset, develop a model:
    - Prepare your data.
    - Pick your evaluation protocol: holdout validation? K-fold validation? Which portion of the data should you use for validation?
    - Achieve statistical power: beat a simple baseline.
    - Scale up: develop a model that can overfit.
    - Regularize your model and tune its hyperparameters, based on performance on the validation data. A lot of machine learning research tends to focus only on this step, but keep the big picture in mind.
  - When your model is ready and yields good performance on the test data, it’s time for deployment:
    - First, make sure you set appropriate expectations with stakeholders.
    - Optimize a final model for inference, and ship a model to the deployment environment of choice—web server, mobile, browser, embedded device, etc.
    - Monitor your model’s performance in production, and keep collecting data so you can develop the next generation of the model 

### Chapter 7: Working With Keras: A Deep Dive
- Learn:
  - Creating Keras models with the Sequential class, the Functional API, and a model subclassing.
  - Using builtin Keras training and evaluation loops.
  - Using Keras callbacks to customize training.
  - Using TensorBoard to monitor training and evaluation metrics.
  - Writing training and evaluation loops from scratch.
- Progressive disclosure of complexity for model building.
![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/progressive_keras.PNG)
- 3 Ways to build Keras Models:
  - Sequential API: similar to Python List. It's limited to simple stacks of layers.
  - Functional API: focuses on graph-like model architecture. It represents a nice mid-point between usability and flexibility. (Most commonly used).
  - Model subclassing: a low-level option where you write everything yourselve from scratch. This is ideal if you want full control over every little thing. Tradeoff: you won't get access to many built-in Keras features, and you will be more at risk of making mistakes.
- Summary:
  - Keras offers a spectrum of different workflows, based on the principle of progressive disclosure of complexity. They all smoothly inter-operate together.
  - You can build models via the Sequential class, via the Functional API, or by subclassing the Model class. Most of the time, you’ll be using the Functional API.
  - The simplest way to train and evaluate a model is via the default fit() and evaluate() methods.  
  - Keras callbacks provide a simple way to monitor models during your call to fit() and automatically take action based on the state of the model.
  - You can also fully take control of what fit() does by overriding the train_step() method.
  - Beyond fit(), you can also write your own training loops entirely from scratch. This is useful for researchers implementing brand-new training algorithms.

### Chapter 8: Introduction To Deep Learing For Computer Vision
- Learn:
  - Understand CNN (convnets).
  - Using data augmentation to mitigate overfitting.
  - Using a pretrain convnet to do feature extracton.
  - Fine-tuning a pretrained convnet
- Introduction to Convnets
  - The fundamental difference between a densely connected layer and a convolutional layer is: Dense layers learn the global patterns in their input feature space (for MNIST, the patterns involving all pixel), whereas Convolutional layers lean local patterns (patterns found in small 2D windows of the inputs).
  - This key characteristic give convnet 2 interesting properties:
      - *The patterns they learn are translation-invariant*: After learnin a certain pattern in a lower-right corner of a picture, a convnet can recognize it anywhere (such as upper-left corner). A densely connected model would have to learn the pattern anew if it appeared at a new location. This makes convnets data-efficient when processing image (because the visual world is fundamentally translation-invariant): They need fewer training samples to lean representations that have generalization power.
      - *They can learn spatial hierachies of patterns*: A first convolution layer will learn small local patterns such as edges, a second convolution layer will learn larger patterns made of features of the first layer,... THis alows convnets to efficiently learn increasingly complex and abstract visual concept, as the visual world is fundamentally spatially hierarchical.
  - The convolution operation:
      -  The convolution operation extracts patches from its input feature map and applies the same transformation to all of these patches, producing an output feature map. This output feature map is still a rank-3 tensor: it has a width and a height. Its depth can be arbitrary, because the output depth is a parameter of the
layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept “presence of a face in the input,” for instance.
      -  In the MNIST example, the first convolution layer takes a feature map of size (28, 28, 1) and outputs a feature map of size (26, 26, 32): it computes 32 filters over its input. Each of these 32 output channels contains a 26 × 26 grid of values, which is a response map of the filter over the input, indicating the response of that filter pattern at different locations in the input.
     ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/response_filter.PNG)
     - That is what the term feature map means: every dimension in the depth axis is a feature (or filter), and the rank-2 tensor output[:, :, n] is the 2D spatial map of the response of this filter over the input.
     - Convolutions are defined by 2 parameters:
         - *Size of the patches extracted from the inputs*: typically 3x3 or 5x5.
         - *Depth of the output feature map*: the number of filters computed by the convolution.
     - Keras: `Conv2D(output_depth, (window_height, window_width))`
     - A convolution works by sliding these windows of 3x3 pr 5x5 over the 3D input feature map, stopping at every possible location, and extracting the 3D patch of surrounding feature `(window_height, window_width, input_depth)`. Each such 3D patch is then transformed into a 1D vector shape `(output_depth,)`, which is done via a tensor product with a leaned weight matrix (called `convolution kernel`, the same kernel is reused across every patch). All of these vectors (one per patch) are then spatially reassembled into a 3D output map shape `(height, width, output_depth)`. Every spatial location in the output feature map corresponds to the same locaiton in the input feature maps (for example, the lowe-right corner of the output contains info about the lower-right corner of the input)
     ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/convolution_viz.PNG)
     - [C4W1L08 Simple Convolutional Network Example](https://www.youtube.com/watch?v=3PyJA9AfwSk&t=442s)
     - [3B1B - But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)
        - Convolution allows you to smooth out the input signal.
        - For image processing, convolution allows to blur the image. For NN, convolution allows to sharpen the features in the image.
     - [Visualizing Convolutional Neural Networks | Layer by Layer](https://www.youtube.com/watch?v=JboZfxUjLSk)
        - ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/single_channel_image.PNG)
          - Says, we have 28x28x1 image and a Conv2D of 3x3x2. This means that we will slide a filter of size 3x3 twice and output the 26x26x2 output.
        - ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/2_channels_image.PNG)
          - Says, we have a 26x26x2 image (or 3 RGB channel image), we need to have a Conv2D of (3x3x4). This meanns that we will slide a filter of size 3x3x2 (2 as the depth of the input channel) 4 time, then stack them to create a 24x24x4 output. Note that when we do 3x3x2 filter convolution, we basically slide thru each input channel twice using the 3x3 filter then add them at every patch. We don't have to initialize the filter depth in Keras, only needed the output depth.
     - Understanding border effects and padding: If you want to get an output feature map with the same spatial dimentions as the input, you can use padding.
     - Understanding convolution strides: This is the distance betweeen 2 successive windows is a parameter of the convolution.
     - In classification models, instead of strides, we tend to use max-pooling to downsample feature maps.
     
  - The max-pooling operation:
      - Role: aggressively downsample feature maps.
      - But why do this? Why not just keep a large feature map?
        - It isn't conducive to learning a spatial hierarchy of features: It's like trying to recognize a digit by only looking at it thru a windows that are 7x7 pixels. We need the features from the last convolution layer to contain info about the totality of the input. By doing this, we make the next conv layer look at an increasingly large window (in terms of the fraction of original input they cover.)
        - The size is too large for small model. This will cause intense overfit.
- Training a convnet from scratch on small dataset
  - In our case, since the model is relatively larger than the dataset, aka we don't have enough data. Thus overfit happens. 
  - Data augmentation randomly transform the image to a believable-looking images. 
  - Goal, at the training time, the model will never see the same exact picture twice. This helps expose the model to more aspects of the data so it can generalize better.
  - The inputs the model sees are still heavily intercorrelated as they come from a small number of original images. We can't proudce new info; we can only mixing existing info. 
  - Similar as Dropout layer, data augmentation layer is inactive during calls of predict() or evaluate().
  - Thanks for data augmentation and dropout, the model overfit much later. This means that the validation loss and accuracy curves are closer to the training ones at a lower and higher rate respectively. Our model generalized well!
  - Training model from scratch is great, but we got so little data (if we got a large amount of data to work, it won't be a problem). Next we should try to use pretrained model to improve accuracy.
 
- Leveraging a pretrained model
  - A common and highly effective approach to deep learning on small image datasets is to use a pretrained model. A pretrained model is a model that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, the spatial heirarchy of features learned by the pretrained model can effectively act as a generic model of the visual world. Thus, its features can prove useful for many different CV problem, even through these new problems may involve completely different classes than those of the original task. 
  - Consider a large convnet trained on the ImageNet dataset (1.4 million labeled image and 1000 different classes).
  - There are 2 ways to use a pretrained model: *feature-extraction* and *fine-tuning.
  - Feature Extraction: Here we have 2 ways
    - Fast Feature Extraction
      - Run the convolutional base over the dataset, record its output to numpy array on disk and then use data as input to a standablone, densely connected classifier similar to those you saw in chapter 4. This solution is fast and cheap to run, becuase it only requires running the convolutional base once for every input image and the convolutional base is by far the most expensive part of the pipeline. However, this technique won't allow us to use data augmentation.
      - Extend the model we have (conv_base) by adding Dense layers on top, and run thw whole thing from end to end on input data. This will allow us to use data augmentation, becuase every input image goes thru the conv_base every time it's seen by the model. But for the same reason, this technique is far more expensive than the first.
    - Feature Extraction with Data Augmentation
      - First, we need to freeze the convolutional base. Freezing a layer or a set of layers means preventing thei weights from being updated during training. If we don't do this, the representations that were previously learned by the convolutional base will be modified during training.
  - Fine Tunning: consist of unfreexing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model and these top layer. It is called fine-tunning because it slightly adjusts the more abstract representations of thte model being reused in order to make them more relevant for the problem at hand. it’s necessary to freeze the convolution base of VGG16 in order to be able to train a randomly initialized classifier on top. For the same reason, it’s only possible to fine-tune the top layers of the convolutional base once the classifier on top has already been trained. If the classifier isn’t already trained, the error signal propagating through the network during training will be too large, and the representations previously learned by the layers being fine-tuned will be destroyed.
    - Steps:
      - Add our custom network on top of an already-trained base network.
      - Freeze the base network.
      - Train the part we added.
      - Unfreeze some layers in the base network.
      - Jointly train both these layers and the part we added.
    - Why not fine-tune more layers? Why not fine-tune the entire convolutional base? You could. But you need to consider the following:
      - Earlier layers in the convolutional base encode more generic, reusable features, whereas layers higher up encode more specialized features. It’s more useful to fine-tune the more specialized features, because these are the ones that need to be repurposed on your new problem. There would be fast-decreasing returns in fine-tuning lower layers.
      - The more parameters you’re training, the more you’re at risk of overfitting. The convolutional base has 15 million parameters, so it would be risky to attempt to train it on your small dataset.
- Summary:
  - Convnets are the best type of machine learning models for computer vision tasks. It’s possible to train one from scratch even on a very small dataset, with decent results.
  - Convnets work by learning a hierarchy of modular patterns and concepts to represent the visual world.
  - On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data.
  - It’s easy to reuse an existing convnet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets.
  - As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further.

### Chapter 9: Advanced Deep Learning For Computer Vision
- Learn:
  - The different branches of CV: image classification, image segmentation, object detection.
  - Modern convnet architecture patterns: residual connection, batch normalization, depthwise separable convolutions.
  - Techniques for visualizing an interpreting what convnet learn

### Architectures (on MNIST)
- [ResNet50]()
- [InceptionV3]()
- [VGG19]()
- [Auto Encoder for Anomaly Detections]
  
### Chapter 10: Deep Learning for Time Series
- Learn: 
  - Examples of ML tasks that involve timeseries data.
  - Understanding RNNs
  - Applying RNNs to a temperature-forecasting example.
  - Advaned RNN usage patterns.
- Different kinds of timeseries tasks.
  - Timeseries can be any data obtained via measurements at regular intervals, like the daily price of a stock, the hourly electricity consumption of a city or the weekly sales of a store. Unlike iid dataa, working with timeseries involves understanding the dynamics of a system - its periodic cycles, how it trends over time, its regular regime and its sudden spike.
  - 3 types of tasks:
    - Classification: assign one or more categorical labels to a time series. Ex: given the timeseries of the activity of a visitor on a website, classify whether the visitor is a bot or a human.
    - Event detection: Identify the occurence of specific epected event within a continuous data stream.
    - Anomaly detection: Detect anything unusual happening within a continuous data stream. Anomaly detection is typically done bia unsupervised learning, because you often don't know what kind of anomaly you are looking for, so you can't train on specific anomaly examples.
 
- A temperature forecasting example
  - Periodicity over multiple timescales is an important and very common property of timeseries data. Whether you’re looking at the weather, mall parking occupancy, traffic to a website, sales of a grocery store, or steps logged in a fitness tracker, you’ll see daily cycles and yearly cycles (human-generated data also tends to feature weekly cycles). When exploring your data, make sure to look for these patterns.
  - In all our experiments, we’ll use the first 50% of the data for training, the following 25% for validation, and the last 25% for testing. When working with timeseries data, it’s important to use validation and test data that is more recent than the training data, because you’re trying to predict the future given the past, not the reverse, and your validation/test splits should reflect that.
  - **Research question**: Given the data cover the previous five days and sampled once per hour, can we predict the temperature in 24 hours?
  - We don't have to vecturize the data as it is numerical. However, since the data is in different sclae, we need to normalize it. WIthin the training data (210,225) timesteps as the training data, we compute the mean and std one the fracture of the data.
  - **Note**: each dataset yields a tuple (samples, targets), where samples is a batch of 256 samples, each contains 120 consecutive hours of input, and targets is the correct corresponding array of 256 target temperatures. Note that the samples are randomly shuffled so 2 consecutive sequences in batch (samples[0] and samples[1] aren't neccesarily close.
  - Todo:
    - Build a common-sense model: assume that the temporal data is continuous: temperature today ~= tomorrow. See what's the MAE.
    - Use Fully dense-layer: Performance not as good. Why? The densely connected approach first flattened the timeseries, which removed the notion of timefrom the input data. 
    - Use CNN: performance worse. Why? The convolutional approach treated every segment of the data in the same way, even applying pooling, which destroyed order information.
      - Weather datase does not has respect for the translation invariance assumption, it only do for a very specific timescale.
      - Order of the data matter a lot: The recent past is far more informative for predicting the next day's temperature than data from 5 days ago. A 1D convnet not able to leverage this fact. In particular, our max pooling and global average pooling largely destroying order information.
    - Use LSTM: Passed the common-sense benchmark.

- Understand RNNs:
  - Dense or Convolutional model have no memory. Each input shown to them is processed independently, with no state kept between inputs. To have theses two success in TS, we have to process a sequence of data into a data point and show them.
  - RNN adopts the principle: it processes sequences by iterating through a sequence eleemnts and maintaining a state that contains info relative to what it has seen so far. RNN is a type of model that has internal loop.
  - The state of the RNN is reset between processing two different, independent sequences (such as two samples in a batch), so you still consider one sequence to be a single data point: a single input to the network. What changes is that this data point is no longer processed in a single step; rather, the network internally loops over sequence elements.

- Advanced use of RNNs
  - Recurrent Dropout: to fight overfitting.
  - Stacking Recurrent layers: increase the representational power of the model.
    - As the model are no longer overfit after regularization, we should increase the capacity of the model.
    - We can either increase the number of units in the layers or adding more layer
  - Bidirectional Recurrent layers: increase accuracy & mitigate forgetting issue.
    - RNN trained on reversed sequences will learn different representations than one trained on the original sequences
    ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/bidirectional_rnn.png)
    
- Summary:
  - As you first learned in chapter 5, when approaching a new problem, it’s good to first establish common-sense baselines for your metric of choice. If you don’t have a baseline to beat, you can’t tell whether you’re making real progress.
  - Try simple models before expensive ones, to make sure the additional expense is justified. Sometimes a simple model will turn out to be your best option.
  - When you have data where ordering matters, and in particular for timeseries data, recurrent networks are a great fit and easily outperform models that first flatten the temporal data. The two essential RNN layers available in Keras are the LSTM layer and the GRU layer.
  - To use dropout with recurrent networks, you should use a time-constant dropout mask and recurrent dropout mask. These are built into Keras recurrent layers, so all you have to do is use the recurrent_dropout arguments of recurrent layers.
  - Stacked RNNs provide more representational power than a single RNN layer. They’re also much more expensive and thus not always worth it. Although they offer clear gains on complex problems (such as machine translation), they may not always be relevant to smaller, simpler problems.
  

### Chapter 11: Deep Learning for Text
- Learn:
  - Preprocessing text data for ML applications.
  - Bag-of-words approaches and sequence-modeling approaches for text processing.
  - The Transformer architecture.
  - Sequence-to-sequence learning.
- NLP About: using ML and large datasets to give computers the ability not to understand language but to ingest a piece of language as input and return something useful:
  - “What’s the topic of this text?” (text classification)
  - “Does this text contain abuse?” (content filtering)
  - “Does this text sound positive or negative?” (sentiment analysis)
  - “What should be the next word in this incomplete sentence?” (language modeling)
  - “How would you say this in German?” (translation)
  - “How would you summarize this article in one paragraph?” (summarization)
  - etc.
- Prepare text data
  - Text vectorization:
    - First, you standardize the text to make it easier to process, such as by converting it to lowercase or removing punctuation.
      - Text standardization: a basic form of feature engineering that aims to erase encoding differences that you don’t want your model to have to deal with.
    - You split the text into units (called tokens), such as characters, words, or groups of words. This is called tokenization.
      - Word-level tokenization.
      - N-gram tokenization.
      - Character-level tokenization.
    - You convert each such token into a numerical vector. This will usually involve first indexing all tokens present in the data.
    ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/text_vectorization.png)
   
- 2 approaches for representing groups of words: Sets and sequences
  - [A Complete Overview of Word Embeddings](https://www.youtube.com/watch?v=5MaWmXwxFNQ&ab_channel=AssemblyAI)
  - There are two approachs: Sets (aka bags of words, unigram or bigram) or Sequences. We will focus on sequence model. By using sequence model, we can remove manual feature engineerning.
  - To do this, you will need to represent the input samples as sequences of integer inddices (1 integer standing for 1 words). Then you will map each integer to a vector obtain ector sequence. Then you will feed these sequences of vector into a stack layers that could cross correlate feature from adjacent vector such as 1D conv net, RNN, Transformer, or bidirectional RNNs and LSTM
  - What is word embedding?

- Beyond text classification: Sequence-to-sequence learning
  - Machine Translation Example
  - Sequence-to-sequence learning with RNN
  - Sequence-to-sequence learning with Transformer
  - How does text-translation for sequence-to-sequence model works?
    - Text generation is text translation.

- Summary (TBD)




### Architectures & DP Specific
- Note: For these sequential model, it depends on the input length (# of days) that we determine the number of time that we unroll the architecture.
![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/lstm_example2.png)

- [RNN Architecture Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80&ab_channel=StatQuestwithJoshStarmer)
![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/rnn_explained.png)
  - For a single RNN, regardless of the number of timesteps, the weights and biased are the same among them.
  - Vanilla RNN use Tanh activation function.
  - [Intuition](https://www.youtube.com/watch?v=LHXXI4-IEns&ab_channel=TheA.I.Hacker-MichaelPhi): 
  - Problem: the more RNN that we unroll, the harder it is to train (Vanishing/Exploding Gradient Problem). How? If we set the weight > 1 then it will increase exponentially. Says if we have 50 dates of sequential data, then we have to unrolll RNN 50 times. This large number will make it hard to take small steps to find the optimal weights and biases during gradient descent. The Gradient can be very large or small which make the learning step bounding a lot. If we make the weight < 1 then the Vanishing Gradient Problem will occur. We also call this problem a short term memory due to backpropagation.
  - How do we solve this? LSTM and GRU (as they have gates to determine which info is added and removed from the hidden state).
  
- [LSTM Architecture](https://www.youtube.com/watch?v=YCzL96nL7j0&ab_channel=StatQuestwithJoshStarmer)
  ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/lstm_viz.png)
  - A type of RNN that avoid vanishing/explosion gradient problem. Instead of using 1 path like an unrolled-RNN, LSTM use 2 paths (1 for long term memory, 1 for short term memory).
  - LSTM uses sigmoid (turn number to [0, 1]) and tanh activation function (turn number to [-1, 1]).
  - Architecture:
  ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/lstm_explained.png)
    - First Stage (blue) - Forget Gate:
      - Green line (Cell-state): represent Long-Term memory. As there is no weight and bias to modify it direclty which allow a series of unrolled units without causing explode or vanish.
      - Pink line (Hidden-state): represent Short-Term memory. This path has weight.
      - The first stage unit determines what percentage of the Long-Term memory is remembered.
    - Second Stage (green + orange) - Input Gate:
      - Orange unit: combines the short terms with the input to create a Potential Long-term memory.
      - Green unit: determines what percentage of the Potential Memory to add to the long-term memory.
      - Here, we will combine with the forget stage to determine the new Long-term memory. 
    - Third Stage: update the Short-Term memory - Output Gate:
      - Pink: combines the long terms with the input to create a Potential Short-term memory.
      - Purple: determines what percentage of the Potential Memory to add to the short-term memory.
    - Thus:
      - we determine what percentage of previous long term to forget
      - combined with the input and long term, we determine what percentage of the previous long term to remember.
      - combine with the input and short term, we determine what percentage of the previous short term to remember
      - Then we will output the new Long-term and Short-term memory.
      ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/lstm_example.png)
    - How? By using a separated path for Long-Term memory and Short-Term memory, LSTM avoid the exploding/vanishing sequence which allow use to put longer input data. LSTM basically saves info fot later, thus preventing older signals from graduallly vanishing during processing. This is similar to residual connection in ResNet. LSTM allows the past info to be reinjected at a later time, thus fighting the vanishing-gradient problem.

- [GRU Architecture](https://www.youtube.com/watch?v=8HyCNIVRbSU&ab_channel=TheA.I.Hacker-MichaelPhi)
    ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/lstm_vs_gru.png)
    - GRU is similar to LSTM. It get rid of the cell state and use hiddne state to transfer information. 
    - GRU has 2 gates:
      - Reset gate: Act similar to Forget gate.
      - Update gate: Act similar to Input gate.

- [Word Embedding](https://www.youtube.com/watch?v=5MaWmXwxFNQ&ab_channel=AssemblyAI)
  - There are one-hot encoding approach or TF-IDF, but these methods does not consider the context of the word in the sentence.
  - Embeddings aim to represent the word in a dense vector, while making sure that the similar words are close to each other in the embedding space. For example, the vector of the word 'tea' and 'coffee' will have a closer distance compared to 'pea'.
  - How are the word embedding made? Learn from a lot of text. We can have a custom embedding layer in the model. The embedding is very specific with the use case.

- [Transformer Architecture](https://www.youtube.com/watch?v=4Bdc55j80l8&t=616s&ab_channel=TheA.I.Hacker-MichaelPhi)
  - The attention mechanism does not suffer from shorterm memory.
  - Transformer is the attention based Encoder-Decoder model.
  - Encoder: output the continuos vector representation of the inputs.
    - Splitted input sentence is fetch into the input embedding layer which create a learned representation of each word. The size of the vector (dimension of embedding space) is depending on the computational power.
    - **Unique***: positional encoding layer: As the transformer does not have any recurrent like RNN, we must add info about the embedding. The author use sine (even timestep) and cosine functions (odd timesteps). The 2 functions is chosen as they allow the model easy to learn.
    - Encoder layer: Map input sequence into an abstract representation vector that encode the info of the input. It contains 2 sub modules: multi-head attention and fully connected network. There also resual connect and Layer normalization.
      - MHA:
        - Summary: It's a module in the encoder that calculate the attention weight for the input and produce the output vector with encoded information on how each word to attend to other words in a sequence
        - Details:: Apply self-attention which allow each word in the input to associate other words in the input. First we input the positional encoding into 3 fully connected layer to create a query, key, value vector. These vector concept comes from data retrieval (when you type a query on youtube, the engine will map query to the key, such as video title, associate with candidate video in the database. then it present you the best matched video). The query and key are multiplied to create a scored matrix. The score matrix emphasize how much the word to focus on other words. Then the matrix is scale down by divided by the dimention of the query and passed thru softmax so that the higher the value, the more attention it is and vice versa. This allow the model to be more confident on which word to attend to. Then the attention weight matrix multiply with the value to provide output. The higher the softmax score will allow the model to focus on that word. Then the output is passed thru a fully connected layer. If we use multi-head, we have to split the embedded input into N part, process attention, then concate prior to linear.
      - Then, the output multiheader attention is added with the original input (residual connection). Then we go thru Layer Normalization. 
      - Then it feed thru the point-wise feed forward network (Linear -> Relu -> Linear). The output is then got residual connection to the output and got layer normalized.
      - The residual helps the model train by allowing gradient to flow thru the network directly. 
      - The Layer Normalization helps the network stabilize the network.
      - The pointwise process the attention by transformed it to a different / better representation.

  - Decoder: takes the continuos representation and step-by-step generate single output while fetched the previous output. The job is to generate text sequences. It has 2 Multi-Header Attention modules, A point-wise feedforward layer and a residual connectionn to a Layer Normalization after each sublayer. The decoder is autoregressive, meaning it takes previous output as input as well as the encoder output that contain the attention info of the input. 
    - For example: "Hello, How are you?" -> "<start> Hola, Como Estas" by generate each groupd of words increasingly.
    - Multi-head self-attention - similar with the mask as the current word can attend to itself and previous word.
    - Multi-head attention: match the encoder output with decoder input which decide which decoder input to put focus on.
    - Then it go thru a point-wise feed forward network for richer representation.
    - The output is a linear classifier -> softmax: The classifier is as big as the nunber of class your have. For example, that you have 10,000 words, then the output size if 10k. It the provide the probability score of each index. We take the largest probability and the index is the word we translate.

- [Batch Normalization](https://www.youtube.com/watch?v=yXOMHOpbon8&t=444s&ab_channel=AssemblyAI)
  - **A way to solve the unstable gradient problem in the neural network, make it train faster, and deal with the overfitting problem at the same time.**
  - Normalzation: Collapse input to be betweeen 0 and 1.
  - Standardization: Change value to make the mean = 0 and variance = 1. Does not mean the value will be inside the 1 bound.
  - If you feed into the network unnormalized data, we might have problem in explosion/vanish gradient.
  - **To Do:** Instead of just normalized our input, we add BatchNorm layer intersection between hidden layer. The input of each hidden layer is standardized, multiply with scale and add with an offset variable. The scale and offset variable are two learning parameters.
  - **Thus**, Batch norm tries to find a good transformation that work for the data point and help the gradient more stable. 
  - **Effect:** Batch normalization allow use to train less epoch. While reduce the need for other regularization

- [Layer Normalization]()

- [Regularization](https://www.youtube.com/watch?v=EehRcPo1M-Q&ab_channel=AssemblyAI)
  - Use regularization to fix overfitting. Overfitting == high variance.
  - Regularization limits the flexibility of the model byy lowering the weights. There are couple of methods to do that:
    - L1/L2 regularization: by adding the weights in the loss calculation to effectively penalize the model with high weights
      - L1 (Lasso) regularization makes the network sparse. To do: add the sum of the absolute values of the weights to the loss. L1 encourage weights to be 0.0, resulting in a more sparse network (weights with more 0.0 values)
      - L2 (ridge regression / weight decay): Add the sum of the squared value of the weight to the loss. L2 penalizes larger weights severely, results in less sparse weights.
    - Dropout: In every training step, each neuron has a probability being inactive.
    - Early Stopping: Stop the learning once the validation error is minimum (once it overfit)
    - Data Augmentation.

- [Bias & Variance for ML]()

- [ML Model Evaluation](https://www.youtube.com/watch?v=LbX4X71-TFI&list=PLcWfeUsAys2nPgh-gYRlexc6xvscdvHqX&index=7&ab_channel=AssemblyAI)
  - Classification:
    - Accuracy = (# of correctly classified instances) / (# of all instances). However, this metric simply evaluation process too much
    ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/prec_recall.png)
    - Precision (Binary Classification) = (TP / (TP + FP)). 
      - Answer: Out of everything I labeled as positive, how many of them belong to that class?
      - When to use? to compare models
    - Recall (Binary Classificationn) = (TP / (TP + FN)). 
      - Answer: Out of everything belong to positive class, how many of them I was able to capture
      - When to use?
    - F1 = combination of precision and recall. 
      - When to use?
    - ROC: Compare TP vs FP.
    - AUC: 
    - Crossentropy: Calculate the distance between the 2 probability distribution.
      - Binary_Crossentropy
      - Crossentropy
  - Regression
    - MAE (Mean Absolution Error)
    - RMSE (Root Mean Squared Error)
    - R^2 (Coefficient of Determination): How well your model fit the data (your model's curve vs actual data)

- [Gated Transformer]()

- [TimeGAN]()

- [Multi-Dimensional Scaling]()

- [Temporal Graph Convolutional Networks (TGCN)]()

- [Transfer Learning](https://www.youtube.com/watch?v=DyPW-994t7w&ab_channel=AssemblyAI)
  - Use the model learned from another task to learn a new task. We use the pretrained model to fine-tune for another task.
  - Why don't we train from scratch? Transfer Learning solved the problem of lacking of data.

### Chapter 12: Generative Deep Learning (X)

### Chapter 13: Best Practices for Real World (Scan Through)
- Ensemble Learning.

### Chapter 14: Conclusion (X)

## Participated Competitions
