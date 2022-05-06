.. SeqIO documentation master file, created by
   sphinx-quickstart on Fri Jun 11 19:57:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SeqIO's documentation!
=================================

SeqIO is a library for processing sequential data to be fed into downstream
sequence models. It uses [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
to create scalable data pipelines but requires minimal use of TensorFlow. In
particular, with one line of code, the returned dataset can be transformed to a
numpy iterator and hence it is fully compatible with other frameworks such as
[JAX](https://github.com/google/jax) or
[PyTorch](https://pytorch.org/).

Currently, SeqIO assumes that the dataset is a sequence, i.e., each feature is
one-dimensional array. Modalities such as text or audio are naturally supported.
Images are supported as long as they are represented as sequences (e.g.,
[Image GPT](http://proceedings.mlr.press/v119/chen20s.html)). We will release this constraint
in the future in order to support higher dimensional data.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   overview.md

.. automodule:: seqio
   :imported-members:
   :members:
   :undoc-members:
   :show-inheritance:

   seqio.beam_utils