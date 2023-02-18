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
    - For imbalance classification, ranking or multilabel classificaiotn, use precision and recall, weighted accuracy, and ROC AUC.

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
     ![](https://github.com/mnguyen0226/kaggle_notebooks/blob/main/docs/imgs/convolution_viz .PNG)
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

- Leveraging a pretrained model


### Chapter 9: Advanced Deep Learning For Computer Vision
- Learn:
  - The different branches of CV: image classification, image segmentation, object detection.
  - Modern convnet architecture patterns: residual connection, batch normalization, depthwise separable convolutions.
  - Techniques for visualizing an interpreting what convnet learn

### Architectures (on MNIST)
- ResNet50
- InceptionV3
- VGG19
- Ensemble Model (pretrained)
  
### Chapter 10: Deep Learning for Time Series
- Learn: 
  - Examples of ML tasks that involve timeseries data.
  - Understanding RNNs
  - Applying RNNs to a temperature-forecasting example.
  - Advaned RNN usage patterns.

### Chapter 11: Deep Learning for Text
- Learn:
  - Preprocessing text data for ML applications.
  - Bag-of-words approaches and sequence-modeling approaches for text processing.
  - The Transformer architecture.
  - Sequence-to-sequence learning.

### Architecture ()
- RNN
- LSTM
- GRU
- Transformer

### Chapter 12: Generative Deep Learning

### Chapter 13: Best Practices for Real World

### Chapter 14: Conclusion

## Participated Competitions
