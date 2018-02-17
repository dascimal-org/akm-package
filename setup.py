from distutils.core import setup
from setuptools import find_packages

install_requires = [
    'numpy',
    'matplotlib',
    'scikit-learn',
    'scipy',
    'dill',
    'pytest',
    'future'
]

setup(
    name='male',
    version='0.1.0',
    url='https://github.com/dascimal-org/akm-package',
    license='MIT',
    author='tund, khanhndk',
    author_email='nguyendinhtu@gmail.com;nkhanh@deakin.edu.au',
    description='MAchine LEarning (MALE) - AKM package',
    packages=find_packages(),
    install_requires=install_requires,
)
