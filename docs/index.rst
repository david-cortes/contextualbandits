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


Other topics
============

Accessing estimator objects from online policies
------------------------------------------------

Online policies generally fit multiple user-supplied estimators. To access the corresponding estimator for a given arm, look at the attribute ``_oracles.algos`` - e.g.:

.. code-block:: python
    
    policy = AdaptiveGreedy(...)
    arm_id = 0
    policy._oracles.algos[arm_id]

In boostrapped estimators:

.. code-block:: python

    policy = BootstrappedUCB(...)
    arm_id = 0
    resample_id = 0
    policy._oracles.algos[arm_id].bs_algos[resample_id]

Serializing (pickling) objects
------------------------------
Don't use ``pickle`` to userialize objects from this package as it's likely to fail. Use ``cloudpickle`` or ``dill`` instead, which have the same syntax and is able to serialize more types of objects.

Using a regression model as the base estimator
----------------------------------------------
The ``base_algorithm`` passed to a policy is normally a binary *classifier* exposing ``predict_proba`` (or ``decision_function``). Several of the policies, however, also accept a plain *regressor* - an estimator with only a ``.predict`` method, such as ``sklearn.linear_model.LinearRegression`` or ``Ridge``. With binary rewards ``r`` in ``{0,1}`` the target conditional mean is ``E[r|x] = P(r=1|x)``, but a regressor fit on the 0/1 labels only *approximates* it: its output is unbounded and can fall below 0 or above 1, so it is not a calibrated probability. The policies below still work because they rank arms by the relative ordering of those estimates (an ``argmax``), not by their absolute value.

``beta_prior`` and ``smoothing`` depend on the *distribution of the rewards*, not on the estimator base: for binary rewards they act on the ``[0,1]`` scale of the reward mean (``smoothing`` shrinks the per-arm score toward a prior rate by observation counts; ``beta_prior`` substitutes ``Beta`` draws in ``(0,1)`` for under-sampled arms). As long as the regressor is unbiased for ``E[r|x]`` its predictions sit on that same scale, so the scores stay comparable and these mechanisms do not need to be disabled for a regressor base. The only caveat is that the regressor's raw output is not a calibrated probability and may fall outside ``[0,1]`` - harmless for the ``argmax``-based policies (which use only the relative ordering), but not safe to read directly as a probability.

Policies that work with a plain regressor base under binary rewards:

* ``BootstrappedUCB`` / ``BootstrappedTS`` - exploration comes from bootstrap resampling of the regressor; arms are ranked by a percentile (UCB) or a sampled estimate (TS) of the predictions.
* ``SeparateClassifiers`` - fits one regressor per arm and picks the ``argmax`` of ``E[r|x]``.
* ``EpsilonGreedy`` - ``argmax`` of the per-arm predictions, with random exploration that is independent of the base.
* ``AdaptiveGreedy`` - keep the default ``decay_type="percentile"``, which calibrates the exploration threshold from the observed score distribution; with a fixed threshold (``decay_type="threshold"`` / ``percentile=None``) the threshold lives on a ``[0,1]`` scale that no longer matches an uncalibrated regressor. The threshold is always positive, so a regressor that predicts negative values tends to explore more than intended.
* ``ExploreFirst`` - works for the default greedy exploitation. The active-sampling option (``prob_active_choice > 0`` with ``f_grad_norm="auto"``) is only supported for a small set of logistic-type estimators and will reject an arbitrary regressor.

Policies that do **not** work with an arbitrary regressor base:

* ``ActiveExplorer`` - its active-learning step needs per-observation gradients; with the default ``f_grad_norm="auto"`` only a fixed set of logistic-type estimators is accepted, so an arbitrary regressor must supply a custom ``f_grad_norm``.
* ``ParametricTS`` - models each arm with a Beta-Binomial posterior whose parameters are ``pred * n`` and ``(1 - pred) * n``, which only makes sense for ``pred`` in ``[0,1]``; an out-of-range regressor output is silently clipped and the arm ends up over- or under-selected.
* ``SoftmaxExplorer`` - runs without error, but it interprets the base output through an inverse-sigmoid and a softmax, so a regressor whose predictions leave ``(0,1)`` is silently saturated. Use it only with a base whose ``predict`` is already bounded to ``(0,1)``.

The built-in linear policies ``LinUCB`` and ``LinTS`` use an internal regression model and take no ``base_algorithm``; ``LogisticUCB`` / ``LogisticTS`` and the ``Partitioned*`` policies likewise use their own built-in models.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
