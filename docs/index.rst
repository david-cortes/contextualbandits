.. Contextual Bandits documentation master file, created by
   sphinx-quickstart on Wed Mar 28 00:01:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contextual Bandits
=================================

This is the documentation page for the python package *contextualbandits*. For 
more details, see the project's GitHub page:

`<https://www.github.com/david-cortes/contextualbandits/>`_

Installation
=================================
Package is available on PyPI, can be installed with
::

    pip install contextualbandits

If it fails to install due to not being able to compile C code, an earlier pure-Python version can be installed with
::

    pip install contextualbandits==0.1.8.5


Getting started
=================================

You can find user guides with detailed examples in the following links:

`Online Contextual Bandits 
<http://nbviewer.jupyter.org/github/david-cortes/contextualbandits/blob/master/
example/online_contextual_bandits.ipynb>`_

`Off policy Learning in Contextual Bandits 
<http://nbviewer.jupyter.org/github/david-cortes/contextualbandits/blob/master/
example/offpolicy_learning.ipynb>`_

`Policy Evaluation in Contextual Bandits 
<http://nbviewer.jupyter.org/github/david-cortes/contextualbandits/blob/master/
example/policy_evaluation.ipynb>`_

Serializing (pickling) objects
=================================
Don't use `pickle` to userialize objects from this package as it's likely to fail. Use `dill` instead, which has the same syntax and is able to serialize more types of objects.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   index


Online Contextual Bandits
=================================

Hint: if in doubt of where to start or which method to choose, the safest bet is `BootstrappedUCB`.

Policy classes - first one from each group is the recommended one to use:

* Randomized:

    * `AdaptiveGreedy <#contextualbandits.online.AdaptiveGreedy>`_
    * `SoftmaxExplorer <#contextualbandits.online.SoftmaxExplorer>`_
    * `EpsilonGreedy <#contextualbandits.online.EpsilonGreedy>`_
    * `ExploreFirst <#contextualbandits.online.ExploreFirst>`_
* Active choices:

    * `ActiveExplorer <#contextualbandits.online.ActiveExplorer>`_
    * `AdaptiveGreedy <#contextualbandits.online.AdaptiveGreedy>`_ (with `active_choice != None`)
    * `ExploreFirst <#contextualbandits.online.ExploreFirst>`_ (with `prob_active_choice > 0`)
* Thompson sampling:

    * `BootstrappedTS <#contextualbandits.online.BootstrappedTS>`_
    * `PartitionedTS <#contextualbandits.online.PartitionedTS>`_
    * `ParametricTS <#contextualbandits.online.ParametricTS>`_
    * `LogisticTS <#contextualbandits.online.LogisticTS>`_
    * `LinTS <#contextualbandits.online.LinTS>`_
* Upper confidence bound:

    * `BootstrappedUCB <#contextualbandits.online.BootstrappedUCB>`_
    * `PartitionedUCB <#contextualbandits.online.PartitionedUCB>`_
    * `LogisticUCB <#contextualbandits.online.LogisticUCB>`_
    * `LinUCB <#contextualbandits.online.LinUCB>`_
* Naive:

    * `SeparateClassifiers <#contextualbandits.online.SeparateClassifiers>`_


ActiveExplorer
--------------
.. autoclass:: contextualbandits.online.ActiveExplorer
    :members:
    :undoc-members:
    :inherited-members:

AdaptiveGreedy
--------------
.. autoclass:: contextualbandits.online.AdaptiveGreedy
    :members:
    :undoc-members:
    :inherited-members:

BootstrappedTS
--------------
.. autoclass:: contextualbandits.online.BootstrappedTS
    :members:
    :undoc-members:
    :inherited-members:

BootstrappedUCB
---------------
.. autoclass:: contextualbandits.online.BootstrappedUCB
    :members:
    :undoc-members:
    :inherited-members:

EpsilonGreedy
-------------
.. autoclass:: contextualbandits.online.EpsilonGreedy
    :members:
    :undoc-members:
    :inherited-members:

ExploreFirst
------------
.. autoclass:: contextualbandits.online.ExploreFirst
    :members:
    :undoc-members:
    :inherited-members:

LinTS
-----
.. autoclass:: contextualbandits.online.LinTS
    :members:
    :undoc-members:
    :inherited-members:

LinUCB
------
.. autoclass:: contextualbandits.online.LinUCB
    :members:
    :undoc-members:
    :inherited-members:

LogisticTS
----------
.. autoclass:: contextualbandits.online.LogisticTS
    :members:
    :undoc-members:
    :inherited-members:

LogisticUCB
-----------
.. autoclass:: contextualbandits.online.LogisticUCB
    :members:
    :undoc-members:
    :inherited-members:

ParametricTS
------------
.. autoclass:: contextualbandits.online.ParametricTS
    :members:
    :undoc-members:
    :inherited-members:

PartitionedTS
-------------
.. autoclass:: contextualbandits.online.PartitionedTS
    :members:
    :undoc-members:
    :inherited-members:

PartitionedUCB
--------------
.. autoclass:: contextualbandits.online.PartitionedUCB
    :members:
    :undoc-members:
    :inherited-members:

SeparateClassifiers
-------------------
.. autoclass:: contextualbandits.online.SeparateClassifiers
    :members:
    :undoc-members:
    :inherited-members:

SoftmaxExplorer
---------------
.. autoclass:: contextualbandits.online.SoftmaxExplorer
    :members:
    :undoc-members:
    :inherited-members:


Off-policy learning
=================================

Hint: if in doubt, use `OffsetTree` or `SeparateClassifiers` (last one is from the online module)

DoublyRobustEstimator
---------------------
.. autoclass:: contextualbandits.offpolicy.DoublyRobustEstimator
    :members:
    :undoc-members:
    :inherited-members:

OffsetTree
----------
.. autoclass:: contextualbandits.offpolicy.OffsetTree
    :members:
    :undoc-members:
    :inherited-members:


Policy Evaluation
=================================

evaluateRejectionSampling
-------------------------
.. autoclass:: contextualbandits.evaluation.evaluateRejectionSampling
    :members:
    :undoc-members:
    :inherited-members:

evaluateDoublyRobust
--------------------
.. autoclass:: contextualbandits.evaluation.evaluateDoublyRobust
    :members:
    :undoc-members:
    :inherited-members:

evaluateFullyLabeled
--------------------
.. autoclass:: contextualbandits.evaluation.evaluateFullyLabeled
    :members:
    :undoc-members:
    :inherited-members:

evaluateNCIS
------------
.. autoclass:: contextualbandits.evaluation.evaluateNCIS
    :members:
    :undoc-members:
    :inherited-members:



Linear Regression
=================================

The package offers non-stochastic linear regression procedures with exact "partial_fit" solutions, which are recommended to use alongside the online policies for better incremental updates.

Linear Regression
-----------------
.. autoclass:: contextualbandits.linreg.LinearRegression
    :members:
    :undoc-members:
    :inherited-members:

ElasticNet
----------
.. autoclass:: contextualbandits.linreg.ElasticNet
    :members:
    :undoc-members:
    :inherited-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
