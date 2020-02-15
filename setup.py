from distutils.core import setup

setup(
  name = 'contextualbandits',
  packages = ['contextualbandits'],
  install_requires=[
   'pandas',
   'numpy',
   'scipy',
   'pandas',
   'joblib>=0.13'
],
  version = '0.1.8.1',
  description = 'Python Implementations of Algorithms for Contextual Bandits',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/contextualbandits',
  keywords = 'contextual bandits offset tree doubly robust policy linucb thompson sampling',
  classifiers = [],
)