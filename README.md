# Handwritten-Digit-Recognition-Using-ML

For this Project, I used the MNIST (Modified National Institute of Standards and Technology) dataset as a source for my training, and testing data. MNIST consists of 70,000 handwritten digits, and 60,000 of them are for training purposes, and 10,000 are for testing purposes.

# How to Set Up Code for this:

Download Jupyter Notebook using the link below:

https://jupyter.org/install

After downloading, set up Jupyter, and proceed with the code provided:

```

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')

X = mnist.data / 255.0
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                      solver='sgd', verbose=10, random_state=1,
                      learning_rate_init=.1)

model.fit(X_train, y_train)

```
