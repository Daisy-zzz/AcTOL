import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='AcTOL',
    py_modules=["actol"],
    version='0.0.0',
    packages=find_packages(),
    description='Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents',
    long_description=read('README.md'),
    install_requires=[
        'torch==1.13.1',
        'torchvision==0.14.1',
        'timm==0.9.12',
        'mmengine',
        'tqdm',
        'numpy',
        'tensorboardX',
        'gdown',
        'openai-clip',
        'chardet'
    ]
)