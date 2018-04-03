# Multinomial Logistic Regression

This repository provides a Multinomial Logistic regression model (*a.k.a* MNL) for the classification problem of multiple classes.

The novelity of this model is that it is implemented with the deep learning framework 'Pytorch'.

A typical scenario to apply MNL model is to predict the choice of customer in a collection of alternatives, which is often referred as Customer Choice Modeling. As a clarification of terminology, the alternative might also be referred as 'option', and the collection of alternatives might be called as 'session'. 


### Organization

- `MNL.py`: this python module contains the implementation of Multinomial Logistic Regression model that is implemented with Pytorch.

- `MNL_aux.py`: this python module provides some auxiliary functions in complement with the `MNL.py` model, e.g. train the model with early stopping on the error delta threshold.


### Usage

One can inspire from the `demo` notebook on the usage of API. In general, one only needs to provides a `dict` of parameters for the training, *e.g.* loss function, optimizer, regularization *etc*.

As to the input data format, any data source that could be transformed into Python dataframe will do. There is only requirement on the content of the data, *i.e.* each data record should contain a `session_id` attribute that group the records/options into a particular choice session.


