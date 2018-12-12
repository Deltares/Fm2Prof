from setuptools import setup, find_packages

# more info: https://docs.python.org/3/distutils/setupscript.html
# https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules

setup(
    name="fm2prof",
    packages=find_packages(exclude=('tests')),
    version='1.0.0',
    description='Functions used for the emulation/reduction of 2D models to 1D models for Delft3D FM (D-Hydro).',
    license='LICENSE.txt',
    long_description=open('README.txt').read(),
    author='K.D. Berends',
    author_email='koen.berends@deltares.nl',
    url='https://www.deltares.nl/nl/',
    download_url='https://repos.deltares.nl/repos/RIVmodels/rivtools/branches/fm2profTool',
    install_requires=[
        "seaborn==0.7.1",
        "pandas==0.18.1",
        "numpy==1.12.1",
        "matplotlib==1.5.1",
        "Django==1.11.5",
        "netCDF4==1.2.9",
        "scipy==1.0.0b1",
        "scikit_learn==0.19.0"
    ],
)