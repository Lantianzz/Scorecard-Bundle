'''
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://choosealicense.com/
'''

import setuptools

with open("README.md", "r" , encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
	name="scorecardbundle",
	version="1.0.0",
	author="Lantian ZHANG",
	author_email="zhanglantian1992@163.com",
	description="An high-level Scorecard modeling API that is Scikit-Learn consistent",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/Lantianzz/Scorecard-Bundle",

	packages=setuptools.find_packages(),
	classifiers=[
				"Development Status :: 4 - Beta",
				"Intended Audience :: Data Analyst",
				"Topic :: Machine Learning :: Scorecard",
	    		"License :: OSI Approved :: BSD 3-Clause License",		
	    		"Programming Language :: Python :: 3"
				],
	keywords='Python Scorecard Modeling',
	packages=find_packages(exclude=['TEMP']),
	python_requires='>=3.5',
	install_requires=['numpy','scipy','pandas','matplotlib','sklearn'],
)

