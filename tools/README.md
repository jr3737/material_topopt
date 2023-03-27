## Mesh file generation for coarse scale geometry
Many free software packages exist for creating a finite element mesh of a general geometry in a format that can be read by [meshio](https://github.com/nschloe/meshio) (see [Gmsh](https://gmsh.info/) for example or even [Delaunay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy-spatial-delaunay) in [SciPy](https://docs.scipy.org/doc/scipy/index.html) for convex shapes). Here, we provide a tool that utilizes a few additional Python libraries (see below for installation details) and simple scripts to create a mesh of a general polygon. The user should start with the simple L-bracket example, copy it, and modify it to fit their needs. This consists primarily of creating a list of vertex coordinates describing the polygon, and a few sets of nodal indices to be used later for applying boundary conditions. The utilities here also write the corresponding finite element mesh out in a suitable format and make a quick plot of the mesh and node sets that the user can quickly check for accuracy.

## Installation

```bash
conda activate material_topopt
conda install -c conda-forge shapely trimesh -y
pip install triangle
```