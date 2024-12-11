# Vertical Federated Learning with Flower Using Scikit-learn Model

This example will showcase how we can perform Vertical Federated Learning with
Flower using model from scikit-learn. We'll be using the [Titanic dataset](https://www.kaggle.com/competitions/titanic/data)
to train simple logistic regression models for binary classification. We will go into
more details below, but the main idea of Vertical Federated Learning is that
each client is holding different feature sets of the same dataset and that the
server is holding the labels of this dataset.

