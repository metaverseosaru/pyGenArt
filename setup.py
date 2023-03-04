from distutils.core import setup

requirements = [
    'numpy',
    'scipy',
    'matplotlib',
    'torch'
]

setup(
    name='pyGenArt',
    version='0.01',
    author='Metaverse Osaru',
    author_email='metaverseosaru@gmail.com',
    scripts=[],
    install_requires=requirements,
    packages=['pyGenArt',],
    package_data={'pyGenArt': ['data/*']},
    include_package_data=True,
    license='LICENSE.txt',
    long_description=open('README.md').read(),
)