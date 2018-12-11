from setuptools import setup, find_packages

# more info: https://docs.python.org/3/distutils/setupscript.html
# https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules

setup(
    name="fm2prof",
    packages=find_packages(),
    version='1.0.0',
    description='Functions used for the emulation/reduction of 2D models to 1D models for Delft3D FM (D-Hydro).',
    author='K.D. Berends',
    author_email='koen.berends@deltares.nl',
    url='https://www.deltares.nl/nl/',
    download_url='https://repos.deltares.nl/repos/RIVmodels/rivtools/branches/fm2profTool')