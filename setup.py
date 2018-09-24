from distutils.core import setup

setup(
  name = 'contextualbandits',
  packages = ['contextualbandits'],
  install_requires=[
   'pandas',
   'numpy',
   'scipy',
   'scikit-learn',
   'costsensitive',
   'pymc3'
],
  version = '0.1.3.1',
  description = 'Python Implementations of Algorithms for Contextual Bandits',
  author = 'David Cortes',
  author_email = 'david.cortes.rivera@gmail.com',
  url = 'https://github.com/david-cortes/contextualbandits',
  download_url = 'https://github.com/david-cortes/contextualbandits/archive/0.1.3.1.tar.gz',
  keywords = 'contextual bandits offset tree doubly robust policy linucb thompson sampling',
  classifiers = [],
)