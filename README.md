# Multinomial Logistic Regression

This repository provides a Multinomial Logistic regression model (*a.k.a* **MNL**) for the classification problem of multiple classes.

The novelity of this model is that it is implemented with the deep learning framework 'Pytorch'.

A typical scenario to apply MNL model is to predict the choice of customer in a collection of alternatives, which is often referred as Customer Choice Modeling. As a clarification of terminology, the alternative might also be referred as 'option', and the collection of alternatives might be called as 'session'. 


### Organization

- `MNL.py`: this python module contains the implementation of Multinomial Logistic Regression model that is implemented with Pytorch.

- `MNL_plus.py`: this python module provides a number of auxiliary functions in complement with the `MNL.py` model. For instance, one can find a train function with the capabililty of early stopping on the predefined threshold of error delta.

- `Mint.py`: this is a minimized model that is intended for inference only, with on dependency on the Pytorch framework. Once one obtains a model with the `MNL` module, one could *export" the trained model to `Mint` and deploy it in the running time with minimal dependencies (panda + numpy).


### Usage

One can inspire from the `demo` notebook on the usage of API. In general, one only needs to provides a `dict` of parameters for the training, *e.g.* loss function, optimizer, regularization *etc*.

As to the input data format, any data source that could be transformed into Python dataframe will do. There is only one requirement on the content of the data, *i.e.* each data record should contain a `session_id` attribute that group the records/options into a particular choice session, and a binary `choice` attribute that indicates whether the option is chosen (value `1`) or not (value `0`).

| session_id |  choice |  feature_1 | feature_2 | feature_n|
| -----------| --------| -----------| ----------| ---------|
|   10001    |    0    |   0.331    | 0.587     |  ...     |
|   10001    |    0    |   0.983    | 0.223     |  ...     |
|   10001    |    1    |   0.732    | 0.945     |  ...     |

