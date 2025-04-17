# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- merge equal poles ([#54](https://github.com/frankebel/DMFT.jl/pull/54)) ([66b6dc2](https://github.com/frankebel/DMFT.jl/commit/66b6dc218e681535645e7434da2ae204ffc4bfd3))

## [0.6.0] - 2025-04-16

### Added

- Green's function and hybridization function on a given grid ([#46](https://github.com/frankebel/DMFT.jl/pull/46)) ([c30556d](https://github.com/frankebel/DMFT.jl/commit/c30556ddc816a1b9cf4aa1436d4a3fa88ce6b3fe))
- subtraction of `Pole` ([#47](https://github.com/frankebel/DMFT.jl/pull/47)) ([8ba021b](https://github.com/frankebel/DMFT.jl/commit/8ba021bee989cf8ba536fd38a51d43711df62775))
- inversion of `Pole` ([#48](https://github.com/frankebel/DMFT.jl/pull/48)) ([c34ff69](https://github.com/frankebel/DMFT.jl/commit/c34ff69349800085824d1430dbd975e83de9e8b3))
- update hybridization function in pole representation ([#50](https://github.com/frankebel/DMFT.jl/pull/50)) ([c0725f5](https://github.com/frankebel/DMFT.jl/commit/c0725f5572155657110980cf8caf28ed130a73cb))
- calculate DMFT self-consistency in pure `Pole` representation ([#52](https://github.com/frankebel/DMFT.jl/pull/52)) ([18c783b](https://github.com/frankebel/DMFT.jl/commit/18c783bfcada4463ebeb77d85fc8bef7fac357c7))

### Changed

- unify function names of Green's function and hybridization function ([#41](https://github.com/frankebel/DMFT.jl/pull/41)) ([d095219](https://github.com/frankebel/DMFT.jl/commit/d095219ade1ae73349ff79e8ea903f69f73159a7)) ([#45](https://github.com/frankebel/DMFT.jl/pull/45)) ([d4a4bf6](https://github.com/frankebel/DMFT.jl/commit/d4a4bf6dda05e8a97f9c749bfef08638d1985f89))
- rename functions ([#49](https://github.com/frankebel/DMFT.jl/pull/49)) ([6fcefff ](https://github.com/frankebel/DMFT.jl/commit/6fcefffa2f80c817b1dfa95a4001cec880ec6b66))
  - `self_energy` → `self_energy_IFG`
  - `self_energy_gauss` → `self_energy_IFG_gauss`
  - `update_weiss_field` → `update_hybridization_function`

### Fixed

- block Lanczos must return hermitian matrices ([#42](https://github.com/frankebel/DMFT.jl/pull/42)) ([d0b066a](https://github.com/frankebel/DMFT.jl/commit/d0b066aba90a8308ea0f9adeece25165e52acaba))
- correlator FR in improved self-energy ([#51](https://github.com/frankebel/DMFT.jl/pull/51)) ([f374b44](https://github.com/frankebel/DMFT.jl/commit/f374b444381c9c2a612561cf0d95c32a1733dff3))

## [0.5.1] - 2025-04-02

### Fixed

- documentation ([#40](https://github.com/frankebel/DMFT.jl/pull/40)) ([032ed29](https://github.com/frankebel/DMFT.jl/commit/032ed2981c1af41a57eb60616dcbab8f40fc8017))

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
