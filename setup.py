from setuptools import setup, find_packages

setup(
    name='nimare-gpu',
    version='0.0.1',
    description='A GPU wrapper for NiMARE',
    author='Amin Saberi', 
    author_email='amnsbr@gmail.com',
    url='https://github.com/amnsbr/nimare-gpu',
    packages=find_packages(),
    install_requires=[
        'NiMARE==0.2.0'
    ],
)