.. SeqIO documentation master file, created by
   sphinx-quickstart on Fri Jun 11 19:57:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

******************************
SeqIO
******************************


.. div:: sd-text-left sd-font-italic

   Task-based datasets, preprocessing, and evaluation for sequence models

----

SeqIO is a library for processing sequential data to be fed into downstream 
sequence models. It uses `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`__ 
to create scalable data pipelines but requires minimal use of TensorFlow. In 
particular, with one line of code, the returned dataset can be transformed to a 
numpy iterator and hence it is fully compatible with other frameworks such as 
`JAX <https://github.com/google/jax>`__ or `PyTorch <https://pytorch.org/>`__.

Installation
^^^^^^^^^^^^

.. code-block:: python

   pip install seqio

Quick Start
^^^^^^^^^^^

Read the SeqIO Guide for a quick introduction to the Task and Mixture APIs, key
underlying components such as the data source, preprocessors, feature converters 
and metrics, and the evaluation API.

`Learn more <overview.html>`__

Tutorials
^^^^^^^^^

Browse through a series of self-contained Colab notebooks that illustrate 
various aspects of defining and running data pipelines using SeqIO Tasks and 
Mixtures, and evaluating models using the SeqIO evaluation library. 

`Learn more <tutorials.html>`__

API Reference
^^^^^^^^^^^^^

Understand the codebase better to write custom components and contribute to the 
effort.

`Learn more <api_reference/index.html>`__

Citing SeqIO
^^^^^^^^^^^^

Please use the following bibtex entry to cite SeqIO.

.. code-block::

   @article{roberts2022t5x,
   url = {https://arxiv.org/abs/2203.17189},
   author = {Roberts, Adam and Chung, Hyung Won and Levskaya, Anselm and Mishra, Gaurav and Bradbury, James and Andor, Daniel and Narang, Sharan and Lester, Brian and Gaffney, Colin and Mohiuddin, Afroz and Hawthorne, Curtis and Lewkowycz, Aitor and Salcianu, Alex and van Zee, Marc and Austin, Jacob and Goodman, Sebastian and Soares, Livio Baldini and Hu, Haitang and Tsvyashchenko, Sasha and Chowdhery, Aakanksha and Bastings, Jasmijn and Bulian, Jannis and Garcia, Xavier and Ni, Jianmo and Chen, Andrew and Kenealy, Kathleen and Clark, Jonathan H. and Lee, Stephan and Garrette, Dan and Lee-Thorp, James and Raffel, Colin and Shazeer, Noam and Ritter, Marvin and Bosma, Maarten and Passos, Alexandre and Maitin-Shepard, Jeremy and Fiedel, Noah and Omernick, Mark and Saeta, Brennan and Sepassi, Ryan and Spiridonov, Alexander and Newlan, Joshua and Gesmundo, Andrea},
   title = {Scaling Up Models and Data with $\texttt{t5x}$ and $\texttt{seqio}$},
   journal={arXiv preprint arXiv:2203.17189},
   year = {2022},
   }

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:
   
   Quick Start <overview>
   Tutorials <tutorials>
   api_reference/index