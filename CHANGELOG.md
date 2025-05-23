# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- merge equal poles ([#54](https://github.com/frankebel/DMFT.jl/pull/54)) ([66b6dc2](https://github.com/frankebel/DMFT.jl/commit/66b6dc218e681535645e7434da2ae204ffc4bfd3))
- logarithmic grid and log-gaussian broadening ([#55](https://github.com/frankebel/DMFT.jl/pull/55)) ([f3a09b0](https://github.com/frankebel/DMFT.jl/commit/f3a09b01e668d3ee188530d1c88ddac1a884b2a6))
- Hubbard-III approximation ([#64](https://github.com/frankebel/DMFT.jl/pull/64)) ([e5dbed7](https://github.com/frankebel/DMFT.jl/commit/e5dbed71f02c0949e646a6ba255a7249c2bf8de5))
- warn if wrong spectral weight exists ([#66](https://github.com/frankebel/DMFT.jl/pull/66)) ([25da3b0](https://github.com/frankebel/DMFT.jl/commit/25da3b0bf829ba4261094e2cbc1d5bad48bb2924))
- merge poles with small weight to neighbors ([#72](https://github.com/frankebel/DMFT.jl/pull/72)) ([7ab44a7](https://github.com/frankebel/DMFT.jl/commit/7ab44a7076250e47c623d3530dc583485d6843c9))
- getters for `Poles` ([#81](https://github.com/frankebel/DMFT.jl/pull/81)) ([add4ec6](https://github.com/frankebel/DMFT.jl/commit/add4ec6d421042265f42b8d9745b059a47fa5254))
- weight(s) of each pole ([#82](https://github.com/frankebel/DMFT.jl/pull/82)) ([b1f1780](https://github.com/frankebel/DMFT.jl/commit/b1f1780830c4a11b3f340cf3a5a64fc6f40e037b))
- moments of `Poles` ([#83](https://github.com/frankebel/DMFT.jl/pull/83)) ([1917ce2](https://github.com/frankebel/DMFT.jl/commit/1917ce246108a72562683d37832b525b1c8e5413))
- similar weight discretization of `Poles` ([#86](https://github.com/frankebel/DMFT.jl/pull/86)) ([0987324](https://github.com/frankebel/DMFT.jl/commit/0987324ceed55c9208cc18c6d10633dc12b129e0))
- `Base.issorted` for `Poles` ([#89](https://github.com/frankebel/DMFT.jl/pull/89)) ([58cd09a](https://github.com/frankebel/DMFT.jl/commit/58cd09a6f8f8fa19578ca9be45a2c1175d90c42a))
- `Base.allunique` for `Poles` ([#90](https://github.com/frankebel/DMFT.jl/pull/90)) ([b3d8e52](https://github.com/frankebel/DMFT.jl/commit/b3d8e52bee8141bc66725e5d5160af5beac5d92d))
- moving poles with negative location to zero ([#91](https://github.com/frankebel/DMFT.jl/pull/91)) ([5dd9f13](https://github.com/frankebel/DMFT.jl/commit/5dd9f13db14f88e19f1aa0bae8aadb9e1208666f))
- flip locations of `Poles` ([#95](https://github.com/frankebel/DMFT.jl/pull/95)) ([5717654](https://github.com/frankebel/DMFT.jl/commit/5717654fc8d59250b719cd43d4ac9ce99b3d8795)) ([#97](https://github.com/frankebel/DMFT.jl/pull/97)) ([b560d33](https://github.com/frankebel/DMFT.jl/commit/b560d330969e40db73357e9a9a727c857c9a5ff1))
- shift locations of `Poles` ([#96](https://github.com/frankebel/DMFT.jl/pull/96)) ([5b85142](https://github.com/frankebel/DMFT.jl/commit/5b85142bb4002ed807d1ad730becf295013fe8b3)) ([#97](https://github.com/frankebel/DMFT.jl/pull/97)) ([0b0ec8f](https://github.com/frankebel/DMFT.jl/commit/0b0ec8f5f7f075e4ccbd566248887f49ab33c220))
- `correlator_plus()`, `correlator_minus()` ([#98](https://github.com/frankebel/DMFT.jl/pull/98)) ([3a78427](https://github.com/frankebel/DMFT.jl/commit/3a78427753859d2a1fc34a9b4af1124be29c1260))
- general block correlator ([#99](https://github.com/frankebel/DMFT.jl/pull/99)) ([4f20faa](https://github.com/frankebel/DMFT.jl/commit/4f20faa45b68c969d47491cb859d25ed7c6856a1))

### Changed

- rename `Pole` → `Poles`, `self_energy_pole` → `self_energy_poles` ([#76](https://github.com/frankebel/DMFT.jl/pull/76)) ([ee98b96](https://github.com/frankebel/DMFT.jl/commit/ee98b96a051d91be21990a2d2f59300735a798b4))
- rename `move_negative_weight_to_neighbors!` → `merge_negative_weight!` ([#77](https://github.com/frankebel/DMFT.jl/pull/77)) ([5d88c89](https://github.com/frankebel/DMFT.jl/commit/5d88c898b2b0507009a78dcacb2f8dac1a36645d))
- rename `merge_equal_poles!` → `merge_degenerate_poles!` ([#79](https://github.com/frankebel/DMFT.jl/pull/79)) ([c5d2b27](https://github.com/frankebel/DMFT.jl/commit/c5d2b27ef19d635e0e93e34912c3b0a04668b2ab))
- remove `to_grid_sqr!` ([#80](https://github.com/frankebel/DMFT.jl/pull/80)) ([cc71c12](https://github.com/frankebel/DMFT.jl/commit/cc71c12923534a4fe5140930780e737b4a790308))
- calculation of self-energy using `Poles` does not enforce original grid ([#84](https://github.com/frankebel/DMFT.jl/pull/84)) ([b976fcd](https://github.com/frankebel/DMFT.jl/commit/b976fcd2b6d1a07b58fe1fb28dc0641d26929e9c))
- calculation of new hybridization function does not enforce original grid ([#85](https://github.com/frankebel/DMFT.jl/pull/85)) ([bba8809](https://github.com/frankebel/DMFT.jl/commit/bba8809f196627f18136f2a40582b9383fe15031))
- merge poles that are `tol` apart ([#88](https://github.com/frankebel/DMFT.jl/pull/88)) ([ad832db](https://github.com/frankebel/DMFT.jl/commit/ad832dbedb84980f8ea0352af543880ee676b7a3))
- merging negative weight function private ([#93](https://github.com/frankebel/DMFT.jl/pull/93)) ([5491f81](https://github.com/frankebel/DMFT.jl/commit/5491f81924d578487557f295cc023f1814e4bcbd))
- remove `update_hybridization_function` for grid ([#102](https://github.com/frankebel/DMFT.jl/pull/102)) ([dc23810](https://github.com/frankebel/DMFT.jl/commit/dc23810bb587498d9dff91bfedbec3a459eda58a))

### Removed

- DMFT step with Lanczos ([#69](https://github.com/frankebel/DMFT.jl/pull/69)) ([34633dd](https://github.com/frankebel/DMFT.jl/commit/34633dd7e88e8e72aad84dbbd496677fd478c434))

### Fixed

- errors in DMFT self-consistency loop ([#57](https://github.com/frankebel/DMFT.jl/pull/57)) ([b10573a](https://github.com/frankebel/DMFT.jl/commit/b10573a5ff1b487878ead85c9fc63c53bc0ed731))
- loss of particle-hole symmetry in natural orbitals ([#58](https://github.com/frankebel/DMFT.jl/issues/58)) ([#59](https://github.com/frankebel/DMFT.jl/pull/59)) ([37ad203](https://github.com/frankebel/DMFT.jl/pull/59/commits/37ad2032a98c06f015ea29152481e9f52333b44c))
- `length` on Pole containing vector and matrix ([#67](https://github.com/frankebel/DMFT.jl/issues/67)) ([9a6dc41](https://github.com/frankebel/DMFT.jl/commit/9a6dc418cbeb3d84a074976c3ad15a0fb997513d))

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
