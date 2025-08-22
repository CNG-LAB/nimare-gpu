from setuptools import setup, find_packages
import versioneer

setup(
    name='nimare-gpu',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A GPU wrapper for NiMARE',
    author='Amin Saberi', 
    author_email='amnsbr@gmail.com',
    url='https://github.com/amnsbr/nimare-gpu',
    packages=find_packages(),
    install_requires=[
        'nimare>=0.2.0',
        'numpy<2'
    ],
)