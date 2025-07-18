# CONTRIBUTING

Contributions are always welcome.
For consistency use the following conventions:

- [Blue style](https://github.com/JuliaDiff/BlueStyle)

- Use lower case `e` for exponential.

  ```jl
  # Yes:
  1e-1
  1e0
  1e1

  # No:
  1E-1
  1E0
  1E1
  ```

- Defining custom structs:
  Each struct gets its own file.
  It should have the following structure:
  - Define the struct.
  - Define constructor(s).
  - Define custom functions in alphabetical order.
  - Define functions from `Core` in alphabetical order.
  - Define functions from `Base` in alphabetical order.
  - Define functions from the standard library (e.g. `LinearAlgebra`) in alphabetical order.
  - Define functions from other packages in alphabetical order.
