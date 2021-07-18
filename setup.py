from setuptools import find_packages
from setuptools import setup

setup(
	name='lyrical_clustering',
	version='0.1',
	packages=find_packages(),
	install_requires=[
		'torch==1.7.0',
		'transformers==3.0.2',
		'tokenizers==0.8.1.rc1',
		'tensorflow==2.3.1',
		'grpcio==1.33.2',
		'google-api-core==1.23.0',
		"tqdm",
		"lxml",
		"undetected_chromedriver",
		"selenium",
		"scikit-image==0.17.2",
		"imutils",
		"opencv-python"
	],
	include_package_data=True,
	description='Lyrics'
)
