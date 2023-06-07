seqio.dataset_providers package
========================

Interfaces
----------

.. currentmodule:: seqio.dataset_providers

.. autoclass:: DataSourceInterface
    :members:

.. autoclass:: DatasetProviderBase
    :members:

.. autoclass:: DatasetProviderRegistry
    :members:

Data Sources
------------

.. currentmodule:: seqio.dataset_providers

.. autoclass:: DataSource
    :members:

.. autoclass:: TfdsDataSource
    :members:

.. autoclass:: FileDataSource
    :members:
    
.. autoclass:: TFExampleDataSource
    :members:

.. autoclass:: TextLineDataSource
    :members:

.. autoclass:: ProtoDataSource
    :members:

.. autoclass:: FunctionDataSource
    :members:

Task
--------

.. currentmodule:: seqio.dataset_providers

.. autoclass:: Task
    :members:

.. autoclass:: ShardInfo
    :members:

.. autoclass:: SourceInfo
    :members:

.. autoclass:: CacheDatasetPlaceholder
    :members:

Mixture
--------

.. currentmodule:: seqio.dataset_providers

.. autoclass:: Mixture
    :members:

.. autoclass:: PyGloveTunableMixture
    :members:

Registry
--------

.. currentmodule:: seqio.dataset_providers

.. autoclass:: TaskRegistry
    :members:

.. autoclass:: MixtureRegistry
    :members:

APIs
----

.. currentmodule:: seqio.dataset_providers
.. autofunction:: get_dataset
.. autofunction:: get_mixture_or_task
.. autofunction:: get_subtasks
.. autofunction:: maybe_get_mixture_or_task
