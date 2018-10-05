from setuptools import setup, Extension

try:
    import numpy as np
except ImportError:
    exit('Please install numpy first.\nUse pip install numpy.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    exit('You need Cython too :(.\n Use pip install cython.\nNo more requirements, promise!')

extensions = [
    Extension(
        'reco.recommender.funksvd',
        ['reco/recommender/funksvd.pyx'],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'reco.recommender.fm',
        ['reco/recommender/fm.pyx'],
        include_dirs=[np.get_include()]
    )
]
ext_modules = cythonize(extensions)

setup(name='reco',
      version='0.1.17',
      description='a simple yet powerful recommender systems library in python',
      url='http://github.com/mayukh18/reco',
      author='Mayukh Bhattacharyya',
      author_email='mayukh.superb@gmail.com',
      license='MIT',
      download_url = 'https://github.com/mayukh18/reco/tarball/0.1.17',
      include_package_data = True,
      keywords=['recommendation'],
      packages=['reco',
                'reco.recommender',
                'reco.datasets',
                'reco.metrics',
                'reco.cross_validation'],
      ext_modules = ext_modules,
      cmdclass= {'build_ext': build_ext},
      install_requires=['numpy',
                        'pandas',
                        ],
      zip_safe=False
      )