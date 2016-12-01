from setuptools import setup

setup(name='reco',
      version='0.1.17',
      description='a simple yet powerful recommender systems library in python',
      url='http://github.com/mayukh18/reco',
      author='Mayukh Bhattacharyya',
      author_email='mayukh.superb@gmail.com',
      license='MIT',
      download_url = 'https://github.com/mayukh18/reco/tarball/0.1.16',
      include_package_data = True,
      keywords=['recommendation'],
      packages=['reco',
                'reco.recommender',
                'reco.datasets',
                'reco.metrics',
                'reco.cross_validation'],
      install_requires=['numpy',
                        'pandas',
                        ],
      zip_safe=False
      )