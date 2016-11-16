from setuptools import setup

setup(name='reco',
      version='0.1.2',
      description='a simple yet powerful recommender systems library in python',
      url='http://github.com/mayukh18/reco',
      author='Mayukh Bhattacharyya',
      author_email='mayukh.superb@gmail.com',
      license='MIT',
      download_url = 'https://github.com/mayukh18/reco/tarball/0.1.2',
      keywords=['recommendation'],
      packages=['reco'],
      install_requires=['numpy',
                        'pandas',
                        'scipy'],
      zip_safe=False
      )