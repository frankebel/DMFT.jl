# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-04-01

### Added

- calculate Kondo temperature ([#35](https://github.com/frankebel/DMFT.jl/pull/35)) ([b1fe8ff](https://github.com/frankebel/DMFT.jl/commit/b1fe8ff94cd895870281b48e2f6a73e0e1c41f7f))
- local Green's function from dispersion relation and optional self-energy ([#36](https://github.com/frankebel/DMFT.jl/pull/36)) ([925c512](https://github.com/frankebel/DMFT.jl/commit/925c51201131ee3fc282848aec5be907628ba789))
- partial Green's function ([#37](https://github.com/frankebel/DMFT.jl/pull/37)) ([1d05c37](https://github.com/frankebel/DMFT.jl/commit/1d05c37dc7556905ef139266a46ecef003360ecb))
- find chemical potential for desired filling ([#38](https://github.com/frankebel/DMFT.jl/pull/38)) ([4547335](https://github.com/frankebel/DMFT.jl/commit/45473351ee71a3a9e736ed836024c35dc97f47ae))
- non-interacting spectral function with Gaussian broadening ([#39](https://github.com/frankebel/DMFT.jl/pull/39)) ([527b4ab](https://github.com/frankebel/DMFT.jl/commit/527b4abfcb29952b322d2b1cb569887cf90fa23d))

### Changed

- rename `Greensfunction` to `Pole` to reflect that it is more generic ([#27](https://github.com/frankebel/DMFT.jl/issues/27)) ([5631a33](https://github.com/frankebel/DMFT.jl/commit/5631a33405a13b292b0b988edf7b14931b59344a))
- unify IO under `read_hdf5`, `write_hdf5` ([#29](https://github.com/frankebel/DMFT.jl/pull/29))
- update `Fermions` dependency to `v0.12.0` ([#30](https://github.com/frankebel/DMFT.jl/pull/30))
- enforce `Pole` pole locations to be on the real axis ([#34](https://github.com/frankebel/DMFT.jl/pull/34)) ([d094356](https://github.com/frankebel/DMFT.jl/commit/d094356cf8502aa6a25cdb0049918715182d459f))
- create module `Debug` and clean up namespace ([#34](https://github.com/frankebel/DMFT.jl/pull/34)) ([c92e6a8](https://github.com/frankebel/DMFT.jl/commit/c92e6a8a33b5bc787028015a5a40012f7e334985))
