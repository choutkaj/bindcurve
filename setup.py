from setuptools import setup, find_packages

with open("README.md", "r") as f: 
    long_description = f.read() 
  
setup( 
    name='bindcurve', 
    version='0.1.0-alpha1', 
    description='A Python package for fitting and plotting of binding curves.', 
    author='choutkaj',
    long_description=long_description, 
    long_description_content_type="text/markdown", 
    packages=find_packages(), 
    install_requires=[ 
        'numpy', 
        'pandas',
        'matplotlib',
        'lmfit',
    ], 
) 