# Changelog

## [0.2.0](https://github.com/CoreySpohn/sinchole/compare/v0.1.0...v0.2.0) (2026-11-19)


### ⚠ BREAKING CHANGES

* use hwoutils Measure instead of measure strings
* stencil(query) on every reader; stencil plots take a reader or profile
* describe grids with hwoutils types and expose the admission rules as a public API

### Features

* **planner:** choose the fewest prefilter taps that meet a tone-error request ([c978e0e](https://github.com/CoreySpohn/sinchole/commit/c978e0ea1e452ac5e4155bda454ca75406ed9cad))
* prefiltered reader keeps coefficients in place of the samples ([a1149c2](https://github.com/CoreySpohn/sinchole/commit/a1149c2a9941823994a560258b585a6969836a4a))
* prepare raster coefficients that leave unsupported source positions dark ([57eb8b3](https://github.com/CoreySpohn/sinchole/commit/57eb8b3e6bc1523caa7b2349014506ad6d368387))
* **reader:** prefiltered kaiser-bessel read profile with coefficients built at load ([3bb6c0d](https://github.com/CoreySpohn/sinchole/commit/3bb6c0d9da264959735ffc8a055bec6a657fef4c))
* scene renderer holds a quarter table and accumulates in float64 or storage dtype ([2e84017](https://github.com/CoreySpohn/sinchole/commit/2e8401765b847d4aa357ab6008c45c738d029a90))
* **scene:** render many sources as one contraction of coefficient images ([36d61eb](https://github.com/CoreySpohn/sinchole/commit/36d61ebf4238f872c0bd97bc7f6a60faf8393720))
* stencil(query) on every reader; stencil plots take a reader or profile ([6df87e7](https://github.com/CoreySpohn/sinchole/commit/6df87e7c3d8764df8a436c2c04c2a7f95de1acf3))


### Bug Fixes

* admit off-frame axis nodes against the image maximum ([f00c017](https://github.com/CoreySpohn/sinchole/commit/f00c017b4dbd3e0dc9315c04aa64f42c0ee30c51))
* **prefilter:** pin the least-squares cutoff and check precision on cached designs ([db94605](https://github.com/CoreySpohn/sinchole/commit/db946059ece9b410ebc71bec9bb5f3bfca0184a8))
* **scene:** contract scene tables in float64 and report only the domain reason for out-of-domain sources ([e5b9c9c](https://github.com/CoreySpohn/sinchole/commit/e5b9c9c233640487afa2edbd85d9d544890d8bf3))


### Code Refactoring

* describe grids with hwoutils types and expose the admission rules as a public API ([2bc8f85](https://github.com/CoreySpohn/sinchole/commit/2bc8f859030c89b52f1517632a2e71eaee550e57))
* use hwoutils Measure instead of measure strings ([0950597](https://github.com/CoreySpohn/sinchole/commit/095059798d9dad30c6c8b40e09068133d40f12d5))

## 0.1.0 (2026-11-06)


### Features

* sinchole 0.1 windowed-sinc grid reader, viz module and docs ([5d20c0e](https://github.com/CoreySpohn/sinchole/commit/5d20c0ee13acb632a6902976d6c24fc7c0240e06))
* **viz:** add plot_stencil_2d and plot_spacing_rule ([fec9fcd](https://github.com/CoreySpohn/sinchole/commit/fec9fcda837c4d3de39918c3cc1e8b86bf678679))
* **viz:** add sinchole.viz plotting module on eyepiece ([3bc9f9a](https://github.com/CoreySpohn/sinchole/commit/3bc9f9ac30f55dacd93e9a20b5f0b3492ebcf8c8))
* windowed-sinc intensity grid admission and reader ([62a840b](https://github.com/CoreySpohn/sinchole/commit/62a840b7f75910ed8da6623049d7a1d9fe4b5fa0))


### Bug Fixes

* enforce symmetry tolerance, numeric range and inclusive coverage at admission ([9f61cd7](https://github.com/CoreySpohn/sinchole/commit/9f61cd7b467d9625245981d1eb1d5d027b553e39))
* enforce the supported domain range when a reader binds a domain ([e9e20f7](https://github.com/CoreySpohn/sinchole/commit/e9e20f7403fe3bb32eab2c71e733f01fd20bc07b))
* **viz:** keep spacing-rule value labels clear of the axis labels ([c2ba650](https://github.com/CoreySpohn/sinchole/commit/c2ba6500cb82a717fa63306aab2b2bbc564b9261))
