from setuptools import setup, find_packages

# more info: https://docs.python.org/3/distutils/setupscript.html
# https://docs.pytest.org/en/latest/goodpractices.html#choosing-a-test-layout-import-rules

setup(
    name="fm2prof",
    packages=(find_packages(exclude=('tests', 'ReportGenerator'))),
    # package_dir={'fm2prof': 'fm2prof'},
    version='1.4.0',
    description='Package use to reduce Delft3D FM models from 2D models to 1D models',
    license='LICENSE.txt',
    long_description=open('README.txt').read(),
    author='K.D. Berends',
    author_email='koen.berends@deltares.nl',
    url='https://www.deltares.nl/nl/',
    download_url='https://repos.deltares.nl/repos/RIVmodels/rivtools/branches/fm2profTool',
    install_requires=[
        "seaborn>=0.7",
        "pandas>=0.18",
        "numpy>=1.12",
        "matplotlib>=1.5",
        "netCDF4>=1.2.9",
        "scipy>=1.0",
        "scikit_learn>=0.19.0",
        "geojson>=2.4.1",
        "shapely>=1.6.4"
    ],
)
