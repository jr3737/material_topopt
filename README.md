## Build and Test Status: ![](https://github.com/jr3737/material_topopt/actions/workflows/test.yml/badge.svg)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?logo=ubuntu\&logoColor=white)](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idruns-on)
[![Mac OS](https://img.shields.io/badge/mac%20os-000000?logo=macos\&logoColor=F0F0F0)](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions#jobsjob_idruns-on)
<h1 align='center'>Material TopOpt</h1>

This code performs material topology optimization of a globally periodic, linear-elastic, bi-material microstructure in order to maximize the stiffness (i.e., minimize the compliance) of a user-specified macroscopic structure subject to user-specified loading. Only 3-node triangular finite elements are implemented for simplicity. The finite element mesh of the macroscopic structure must be supplied in a format that is readable by [meshio](https://github.com/nschloe/meshio), in addition to nodesets (i.e., point sets in [meshio](https://github.com/nschloe/meshio)'s terminology) where boundary conditions will be specified.

## Installation
First, install the [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) python package manager.
```bash
conda create --name material_topopt -y
conda activate material_topopt
conda install python=3.10 -y
conda install numpy scipy matplotlib pandas pytest pytest-cov sphinx spyder -y
conda install -c conda-forge meshio pypardiso shapely trimesh -y
conda update --all -y
conda clean -a -y
pip install triangle
```

Then, if you are on a Linux/Mac system with `git` installed, clone this repository,
```bash
git clone https://github.com/jr3737/material_topopt.git
```
otherwise, download the source as a zip file and unzip the contents.

## Documentation
After cloning this repository, `cd` into the `docs` subdirectory and build the documentation like,
```bash
make html
```
after which you should be able to open the HTML documentation in a web browser by opening the file in "docs/_build/html/index.html"

## Example Usage
In the root directory there is an example script named "SetupDesignProblemAndRun.py" which currently serves as a starting point for using this software. Eventually, the code may be generalized and uploaded as a PyPI package.

## See also
Note that this software also makes use of the MMA submodule provided by [GCMMA-MMA-Python](https://github.com/arjendeetman/GCMMA-MMA-Python).

## License
[MIT license](LICENSE)
