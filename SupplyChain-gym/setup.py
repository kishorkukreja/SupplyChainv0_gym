from setuptools import setup, find_packages
import sys
import os

setup(name="SupplyChain_gym",
      version="1.0.0",
      install_requires=[
		'gym>=0.15.0',
		'numpy>=1.16.1',
		'scipy>=1.0',
		'matplotlib>=3.1',
		'networkx>=2.3'],
      python_requires='>=3.5',
      packages=find_packages(),
      zip_safe=False,
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Developers',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
      ]
)  
