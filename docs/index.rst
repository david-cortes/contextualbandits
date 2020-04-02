.. Contextual Bandits documentation master file, created by
   sphinx-quickstart on Wed Mar 28 00:01:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Contextual Bandits
=================================

This is the documentation page for the python package *contextualbandits*. For 
more details, see the project's home page:

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


Online Contextual Bandits
=================================

.. automodule:: contextualbandits.online
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Off-policy learning
=================================

.. automodule:: contextualbandits.offpolicy
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Policy Evaluation
=================================

.. automodule:: contextualbandits.evaluation
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Linear Regression
=================================

This linear regression class keeps the matrices used for the closed-form solution, so that it can be fit incrementally while giving the same solution as if fitted to all data at once (as opposed to stochastic methods which don't have such property). Ideal for the online methods of this package when using them with streaming data.

.. automodule:: contextualbandits.linreg
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
