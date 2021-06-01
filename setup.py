from setuptools import setup

setup(
    name='clothoids',
    version='0.1',
    description='Methods for generating clothoids.',
    url='https://github.com/cmower/clothoids',
    author='Christopher E. Mower',
    author_email='chris.mower@ed.ac.uk',
    license='BSD 2-Clause License',
    packages=['clothoids'],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
)
