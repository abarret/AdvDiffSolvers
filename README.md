# AdvDiffSolvers (ADS)

AdvDiffSolvers is a C++/Fortran library for solving advection–diffusion equations
in complex domains with general Robin boundary conditions. It extends
[IBAMR](https://ibamr.github.io/) with methods for transporting fluid phase
concentrations, evolving concentrations on immersed surfaces, and coupling the
two through boundary reactions.

The package provides:

- Hybrid cut-cell and semi-Lagrangian advection–diffusion methods; see the
  [associated paper](https://doi.org/10.1016/j.jcp.2021.110805).
- Radial basis function (RBF), Lagrange, and spline reconstruction operators
  for semi-Lagrangian transport.
- Level-set geometry, cut-cell diffusion operators, and boundary reconstruction
  utilities.
- Finite element surface mappings and surface/fluid concentration coupling.
- Example applications and regression tests.

ADS builds both 2D and 3D libraries. It is intended for applications built on
IBAMR, with public headers in [`include/ADS`](include/ADS).

## Getting started

You need CMake 3.15 or newer, compatible C/C++/Fortran compilers, `m4`, and a
**CMake installation of IBAMR 0.18 or newer** with both 2D and 3D libraries and
libMesh support. Use the same compiler and MPI environment as your IBAMR build.

Build with a compatible IBAMR installation and the MPI compiler wrappers used
for that installation. From the repository root:

```sh
cmake -S . -B /tmp/ads-build \
  -DIBAMR_ROOT=/path/to/ibamr/install \
  -DCMAKE_INSTALL_PREFIX=/path/to/ADS/install \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpifort
cmake --build /tmp/ads-build --parallel 4
cmake --install /tmp/ads-build
```

Replace the dependency and installation paths with your own. The build directory
can be any directory outside the source tree; `/tmp/ads-build` is an example.
The default build compiles `ADS2d` and `ADS3d`. Installation copies the libraries,
headers, and CMake package files to `CMAKE_INSTALL_PREFIX`.

Use `-DCMAKE_BUILD_TYPE=Release` for an optimized build, matching the IBAMR
installation. Shared libraries are built by default; use
`-DBUILD_SHARED_LIBS=OFF` for static libraries.

For dependency discovery, compiler selection, examples, tests, troubleshooting,
and linking your own application, see the
[build and installation guide](docs/building.md).

## Try an example

Examples are built explicitly, separately from the default library build:

```sh
cmake --build /tmp/ads-build --target rotational --parallel 4
cd /tmp/ads-build/examples/rotational
./main2d input2d
```

Run from the example's build directory so it can find its input files. To build
all examples, use `cmake --build /tmp/ads-build --target examples --parallel 4`.

## Documentation and source layout

- [Build, installation, and application integration](docs/building.md)
- [`include/ADS`](include/ADS): public classes and API comments
- [`src`](src): integrators, solvers, reconstruction operators, and utilities
- [`examples`](examples): complete applications and input files
- [`tests`](tests): regression programs and reference results
