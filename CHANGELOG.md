# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [0.18.0](https://github.com/rickstaa/stable-gym/compare/v0.17.1...v0.18.0) (2023-07-27)


### Features

* add reference info to step/reset return ([#275](https://github.com/rickstaa/stable-gym/issues/275)) ([7d715bd](https://github.com/rickstaa/stable-gym/commit/7d715bdcbe5fff797cde70a6a4ba621e64cd786e))
* **classicalcontrol:** add additional info to step/reset return ([#274](https://github.com/rickstaa/stable-gym/issues/274)) ([021d846](https://github.com/rickstaa/stable-gym/commit/021d846af2422230c4481e228e0d1810daae6153))
* **fetchreachcost:** add reference info to step/reset return ([#277](https://github.com/rickstaa/stable-gym/issues/277)) ([20eb94d](https://github.com/rickstaa/stable-gym/commit/20eb94d6f0e452f791a1a48e021dad5eba8299f4))
* **minitaur:** add minitaur step/reset reference info return ([be01019](https://github.com/rickstaa/stable-gym/commit/be010196f605ad2b65639de8a02a5215e0135307))
* **minitaur:** add minitaur step/reset reference info return ([#276](https://github.com/rickstaa/stable-gym/issues/276)) ([bdde77c](https://github.com/rickstaa/stable-gym/commit/bdde77caaf9f3668eee73b5fdd27d06962722983))
* **quadxhover:** add extra information to step/reset info ([#272](https://github.com/rickstaa/stable-gym/issues/272)) ([91b979d](https://github.com/rickstaa/stable-gym/commit/91b979db8a6350e35631dd4000155b71e3d9b44a))


### Bug Fixes

* **quadxwaypointscost:** fix directional penalty and add extra step info keys ([#269](https://github.com/rickstaa/stable-gym/issues/269)) ([0afc098](https://github.com/rickstaa/stable-gym/commit/0afc0985fd0357c48fa43b9ff9ba29240a1f324a))


### Documentation

* add missing step return info dict keys ([#278](https://github.com/rickstaa/stable-gym/issues/278)) ([016695f](https://github.com/rickstaa/stable-gym/commit/016695fef7cbf4f5f71bda4424fef70287ff0a20))

## [0.17.1](https://github.com/rickstaa/stable-gym/compare/v0.17.0...v0.17.1) (2023-07-25)


### Bug Fixes

* add tempory PyFlyt RuntimeWarning fix ([#265](https://github.com/rickstaa/stable-gym/issues/265)) ([432f857](https://github.com/rickstaa/stable-gym/commit/432f85758e84cbf33bf15edf7287357fc0b7ea30))

## [0.17.0](https://github.com/rickstaa/stable-gym/compare/v0.16.2...v0.17.0) (2023-07-25)


### Features

* **quadxtrackingcost:** add extra keys to environment step info dictionary ([#263](https://github.com/rickstaa/stable-gym/issues/263)) ([9743330](https://github.com/rickstaa/stable-gym/commit/974333037454fe9667691ec489ae716602e20fdb))

## [0.16.2](https://github.com/rickstaa/stable-gym/compare/v0.16.1...v0.16.2) (2023-07-25)


### Documentation

* fix broken numpy links ([#259](https://github.com/rickstaa/stable-gym/issues/259)) ([20a5f45](https://github.com/rickstaa/stable-gym/commit/20a5f45d4d04f9087e8ad2d30733bbcbb06294dd))

## [0.16.1](https://github.com/rickstaa/stable-gym/compare/v0.16.0...v0.16.1) (2023-07-25)


### Documentation

* update quadx docs ([#257](https://github.com/rickstaa/stable-gym/issues/257)) ([8f4745b](https://github.com/rickstaa/stable-gym/commit/8f4745b856b48321fc2220c636ed2727b3fcb4e4))

## [0.16.0](https://github.com/rickstaa/stable-gym/compare/v0.15.1...v0.16.0) (2023-07-24)


### Features

* add new quadcopter cost environments ([#246](https://github.com/rickstaa/stable-gym/issues/246)) ([382c9fd](https://github.com/rickstaa/stable-gym/commit/382c9fd9edf831e7ef68f8259475175dad20d846))

## [0.15.1](https://github.com/rickstaa/stable-gym/compare/v0.15.0...v0.15.1) (2023-07-21)


### Documentation

* improve cartpole docs ([#242](https://github.com/rickstaa/stable-gym/issues/242)) ([0519d69](https://github.com/rickstaa/stable-gym/commit/0519d69b4444a978e67e45a86a8e6172a441ce33))

## [0.15.0](https://github.com/rickstaa/stable-gym/compare/v0.14.0...v0.15.0) (2023-07-21)


### Features

* **oscillator:** simplify oscillator environments ([#231](https://github.com/rickstaa/stable-gym/issues/231)) ([da9b589](https://github.com/rickstaa/stable-gym/commit/da9b589fb1950a8cee19b934b5bf6f1571016dc3))


### Bug Fixes

* fix oscillator reference formula ([#227](https://github.com/rickstaa/stable-gym/issues/227)) ([c3011b4](https://github.com/rickstaa/stable-gym/commit/c3011b4f11c54831d1c2b52eeed2b0f912ae0d39))


### Documentation

* fix markdown admonitions ([#232](https://github.com/rickstaa/stable-gym/issues/232)) ([fb8ab33](https://github.com/rickstaa/stable-gym/commit/fb8ab335ae133cd9d07b29946a8d192a9f757e22))
* update admonitions to new GFM specification ([5905366](https://github.com/rickstaa/stable-gym/commit/5905366154abf2af84e0ec6464ae35ff9faa4dc1))

## [0.14.0](https://github.com/rickstaa/stable-gym/compare/v0.13.1...v0.14.0) (2023-07-18)


### Features

* add 'MinitaurCost' environment ([#222](https://github.com/rickstaa/stable-gym/issues/222)) ([0f787f5](https://github.com/rickstaa/stable-gym/commit/0f787f5784d6ba1def7f8103088498beacfbbca3))
* add velocity randomize and exclude reference arguments ([#215](https://github.com/rickstaa/stable-gym/issues/215)) ([c32b4ee](https://github.com/rickstaa/stable-gym/commit/c32b4ee6fe80afcd9992ff2937c91e745de3285f))
* fix 'reward_range' name and space accuracy ([#220](https://github.com/rickstaa/stable-gym/issues/220)) ([d11416e](https://github.com/rickstaa/stable-gym/commit/d11416eb157c046048c19856db884fb94c4a4dda))
* increase spaces accuracy to float64 ([#221](https://github.com/rickstaa/stable-gym/issues/221)) ([fd212c6](https://github.com/rickstaa/stable-gym/commit/fd212c6c7cc2a034546752ca4ab1d9b028f56203))


### Bug Fixes

* **minitaurcost:** fix 'MinitaurCost' health penalty calculation ([#225](https://github.com/rickstaa/stable-gym/issues/225)) ([345d8a4](https://github.com/rickstaa/stable-gym/commit/345d8a43d9598edecc8c407836e906dcbd87fc23))

## [0.13.1](https://github.com/rickstaa/stable-gym/compare/v0.13.0...v0.13.1) (2023-07-15)


### Documentation

* improve code API documentation ([#206](https://github.com/rickstaa/stable-gym/issues/206)) ([e866c95](https://github.com/rickstaa/stable-gym/commit/e866c9563d4ae5b2830edcc6e3a312de0cd545c8))

## [0.13.0](https://github.com/rickstaa/stable-gym/compare/v0.12.0...v0.13.0) (2023-07-13)


### Features

* add 'FetchReachCost' environment ([#204](https://github.com/rickstaa/stable-gym/issues/204)) ([69d15e7](https://github.com/rickstaa/stable-gym/commit/69d15e78e1f0656c44a33aff2df30b2f2a4d0a2c))


### Documentation

* **fetchreachcost:** fix math equations ([7a3b485](https://github.com/rickstaa/stable-gym/commit/7a3b485dacb941b523a403d17b604908f72c83c9))

## [0.12.0](https://github.com/rickstaa/stable-gym/compare/v0.11.0...v0.12.0) (2023-07-12)


### Features

* improve env '__main__' plots ([#201](https://github.com/rickstaa/stable-gym/issues/201)) ([fd9a84d](https://github.com/rickstaa/stable-gym/commit/fd9a84da446e12fa91b20968e7659908bfc43f60))

## [0.11.0](https://github.com/rickstaa/stable-gym/compare/v0.10.2...v0.11.0) (2023-07-11)


### Features

* add mujoco envs health penalty ([#197](https://github.com/rickstaa/stable-gym/issues/197)) ([4065435](https://github.com/rickstaa/stable-gym/commit/406543588df94224117e5aa5f0356c5f3657fafa))

## [0.10.2](https://github.com/rickstaa/stable-gym/compare/v0.10.1...v0.10.2) (2023-07-11)


### Bug Fixes

* **swimmercost:** fix 'SwimmerCost' unpickle bug ([#195](https://github.com/rickstaa/stable-gym/issues/195)) ([47d2d48](https://github.com/rickstaa/stable-gym/commit/47d2d48b3bc07fa5331573273c350cd745a503dd))


### Documentation

* improve 'CartPoleCost' documentation. ([#193](https://github.com/rickstaa/stable-gym/issues/193)) ([16e3236](https://github.com/rickstaa/stable-gym/commit/16e323679cd0c5c62a5a892ffd98ff1171dcce05))

## [0.10.1](https://github.com/rickstaa/stable-gym/compare/v0.10.0...v0.10.1) (2023-07-10)


### Documentation

* add snapshot comment to CONTRIBUTING.md ([a26a536](https://github.com/rickstaa/stable-gym/commit/a26a536d2335a25c1fe6bc2d16ff3d318ce61b4a))
* improve ROS envs documentation ([2e0fb54](https://github.com/rickstaa/stable-gym/commit/2e0fb54e354f496899c4386291aec86250a21d40))

## [0.10.0](https://github.com/rickstaa/stable-gym/compare/v0.9.1...v0.10.0) (2023-07-10)


### Features

* add 'HopperCost' environment ([#187](https://github.com/rickstaa/stable-gym/issues/187)) ([eb3c363](https://github.com/rickstaa/stable-gym/commit/eb3c363848faee26da8116d4500d817c9d4db113))
* add 'Walker2dCost' environment ([#190](https://github.com/rickstaa/stable-gym/issues/190)) ([e856d47](https://github.com/rickstaa/stable-gym/commit/e856d472c7614f44e16a829a2538a1060f06fd15))

## [0.9.1](https://github.com/rickstaa/stable-gym/compare/v0.9.0...v0.9.1) (2023-07-10)


### Bug Fixes

* fix unresolved merge conflicts ([#185](https://github.com/rickstaa/stable-gym/issues/185)) ([b52bd75](https://github.com/rickstaa/stable-gym/commit/b52bd751f83ed79ab742bfc29e40da46f87d916f))

## [0.9.0](https://github.com/rickstaa/stable-gym/compare/v0.8.0...v0.9.0) (2023-07-10)


### Features

* add 'HalfCheetahCost' environment ([#182](https://github.com/rickstaa/stable-gym/issues/182)) ([8d0d666](https://github.com/rickstaa/stable-gym/commit/8d0d66621acac98cccee48276a3767144a4011af))

## [0.8.0](https://github.com/rickstaa/stable-gym/compare/v0.7.0...v0.8.0) (2023-07-09)


### Features

* add SwimmerCost environment ([#180](https://github.com/rickstaa/stable-gym/issues/180)) ([f9eb341](https://github.com/rickstaa/stable-gym/commit/f9eb34101fb1cbba2e1080f6a19a1866afcca739))

## [0.7.0](https://github.com/rickstaa/stable-gym/compare/v0.6.48...v0.7.0) (2023-07-08)


### Features

* add complicated oscillator environment ([#178](https://github.com/rickstaa/stable-gym/issues/178)) ([7f1a0d1](https://github.com/rickstaa/stable-gym/commit/7f1a0d1436dd7357e529272a987d1928807337b4))

## [0.6.48](https://github.com/rickstaa/stable-gym/compare/v0.6.47...v0.6.48) (2023-07-04)


### Documentation

* update BLC to SLC ([#176](https://github.com/rickstaa/stable-gym/issues/176)) ([327b475](https://github.com/rickstaa/stable-gym/commit/327b475d63f65fe9f696c8a04080b5619c3a51a1))

## [0.6.47](https://github.com/rickstaa/stable-gym/compare/v0.6.46...v0.6.47) (2023-06-30)


### Documentation

* improve docs ([#172](https://github.com/rickstaa/stable-gym/issues/172)) ([9819d85](https://github.com/rickstaa/stable-gym/commit/9819d851ed66ac08fa15ff7cfc5df71abd79e224))
* set pygment style ([40efaee](https://github.com/rickstaa/stable-gym/commit/40efaee434d6e6c1cd6e2496d6480e100a2d9838))

## [0.6.46](https://github.com/rickstaa/stable-gym/compare/v0.6.45...v0.6.46) (2023-06-29)


### Documentation

* fix urls ([#169](https://github.com/rickstaa/stable-gym/issues/169)) ([aaa5b75](https://github.com/rickstaa/stable-gym/commit/aaa5b75e6d92c69235ff80117984763f280b56d1))

## [0.6.45](https://github.com/rickstaa/stable-gym/compare/v0.6.44...v0.6.45) (2023-06-28)


### Documentation

* fix syntax ([#166](https://github.com/rickstaa/stable-gym/issues/166)) ([07b5e94](https://github.com/rickstaa/stable-gym/commit/07b5e9403fa196fcc689fee4b312abdba263b885))

## [0.6.44](https://github.com/rickstaa/stable-gym/compare/v0.6.43...v0.6.44) (2023-06-23)


### Documentation

* fix math ([#162](https://github.com/rickstaa/stable-gym/issues/162)) ([3b78238](https://github.com/rickstaa/stable-gym/commit/3b782389cb029e4a73e27ea6e9457949baec949e))

## [0.6.43](https://github.com/rickstaa/stable-gym/compare/v0.6.42...v0.6.43) (2023-06-21)


### Documentation

* fix small typo ([14a26ea](https://github.com/rickstaa/stable-gym/commit/14a26ea5679a764ccefe70340eb99878c208f29f))

## [0.6.42](https://github.com/rickstaa/stable-gym/compare/v0.6.41...v0.6.42) (2023-06-19)


### Documentation

* fix github edit page link ([fa972d4](https://github.com/rickstaa/stable-gym/commit/fa972d40b7a8c468c2ca456c34f886463396f7f4))

## [0.6.41](https://github.com/rickstaa/stable-gym/compare/v0.6.40...v0.6.41) (2023-06-19)


### Documentation

* fix edit on github button ([7ecece9](https://github.com/rickstaa/stable-gym/commit/7ecece94f9d24c11350cf2083eae6d1e99f88214))
* fix edit on github button ([#154](https://github.com/rickstaa/stable-gym/issues/154)) ([63d9bdb](https://github.com/rickstaa/stable-gym/commit/63d9bdb3c3f3ee04a7ea967b43c4b7f59d2ef838))

## [0.6.40](https://github.com/rickstaa/stable-gym/compare/v0.6.39...v0.6.40) (2023-06-15)


### Documentation

* add codeconv badge ([#148](https://github.com/rickstaa/stable-gym/issues/148)) ([3dd995e](https://github.com/rickstaa/stable-gym/commit/3dd995e7ab439efaf27cbd7f70716f6ba2fdc470))

## [0.6.39](https://github.com/rickstaa/stable-gym/compare/v0.6.38...v0.6.39) (2023-06-14)


### Documentation

* update docs ([f958cc0](https://github.com/rickstaa/stable-gym/commit/f958cc0c262cbc91fc31e11b318039307816c0a6))
* update documentation ([a4e45ed](https://github.com/rickstaa/stable-gym/commit/a4e45ed995f99992b521f2124af33b07bb8688f9))

## [0.6.38](https://github.com/rickstaa/stable-gym/compare/v0.6.37...v0.6.38) (2023-06-14)


### Bug Fixes

* fix release action naming ([3448955](https://github.com/rickstaa/stable-gym/commit/3448955dfc6f54363dec60707d6223e9a89fa2e1))

## [0.6.37](https://github.com/rickstaa/stable-gym/compare/v0.6.36...v0.6.37) (2023-06-14)


### Documentation

* update contribution release guide ([#128](https://github.com/rickstaa/stable-gym/issues/128)) ([6446f27](https://github.com/rickstaa/stable-gym/commit/6446f27e9c392ce5e908179a537797c6fc1702e2))

### [0.6.36](https://github.com/rickstaa/stable-gym/compare/v0.6.35...v0.6.36) (2023-06-14)

### [0.6.35](https://github.com/rickstaa/stable-gym/compare/v0.6.34...v0.6.35) (2023-06-14)

### [0.6.34](https://github.com/rickstaa/stable-gym/compare/v0.6.33...v0.6.34) (2023-06-13)

### [0.6.33](https://github.com/rickstaa/stable-gym/compare/v0.6.32...v0.6.33) (2023-06-13)

### [0.6.32](https://github.com/rickstaa/stable-gym/compare/v0.6.31...v0.6.32) (2023-06-10)

### [0.6.31](https://github.com/rickstaa/stable-gym/compare/v0.6.30...v0.6.31) (2023-06-10)

### [0.6.30](https://github.com/rickstaa/stable-gym/compare/v0.6.29...v0.6.30) (2023-06-10)

### [0.6.29](https://github.com/rickstaa/stable-gym/compare/v0.6.28...v0.6.29) (2023-06-10)

### [0.6.28](https://github.com/rickstaa/stable-gym/compare/v0.6.27...v0.6.28) (2023-06-10)

### [0.6.27](https://github.com/rickstaa/stable-gym/compare/v0.6.26...v0.6.27) (2023-06-10)

### [0.6.25](https://github.com/rickstaa/stable-gym/compare/v0.6.24...v0.6.25) (2023-06-09)

### [0.6.24](https://github.com/rickstaa/stable-gym/compare/v0.6.23...v0.6.24) (2023-06-09)

### [0.6.23](https://github.com/rickstaa/stable-gym/compare/v0.6.22...v0.6.23) (2023-06-09)

### [0.6.22](https://github.com/rickstaa/stable-gym/compare/v0.6.21...v0.6.22) (2023-06-09)

### [0.6.21](https://github.com/rickstaa/stable-gym/compare/v0.6.20...v0.6.21) (2023-06-09)

### [0.6.20](https://github.com/rickstaa/stable-gym/compare/v0.6.19...v0.6.20) (2023-06-09)

### [0.6.19](https://github.com/rickstaa/stable-gym/compare/v0.6.18...v0.6.19) (2023-06-09)

### [0.6.18](https://github.com/rickstaa/stable-gym/compare/v0.6.17...v0.6.18) (2023-06-09)

### [0.6.17](https://github.com/rickstaa/stable-gym/compare/v0.6.16...v0.6.17) (2023-06-09)

### [0.6.16](https://github.com/rickstaa/stable-gym/compare/v0.6.15...v0.6.16) (2023-06-09)

### [0.6.15](https://github.com/rickstaa/stable-gym/compare/v0.6.14...v0.6.15) (2023-06-09)

### [0.6.14](https://github.com/rickstaa/stable-gym/compare/v0.6.13...v0.6.14) (2023-06-06)

### [0.6.13](https://github.com/rickstaa/stable-gym/compare/v0.6.12...v0.6.13) (2023-06-06)

### [0.6.12](https://github.com/rickstaa/stable-gym/compare/v0.6.11...v0.6.12) (2023-06-06)

### [0.6.11](https://github.com/rickstaa/stable-gym/compare/v0.6.10...v0.6.11) (2023-06-06)

### [0.6.10](https://github.com/rickstaa/stable-gym/compare/v0.6.9...v0.6.10) (2023-06-06)

### [0.6.9](https://github.com/rickstaa/stable-gym/compare/v0.6.8...v0.6.9) (2023-06-06)

### [0.6.8](https://github.com/rickstaa/stable-gym/compare/v0.6.7...v0.6.8) (2023-06-06)

### [0.6.7](https://github.com/rickstaa/stable-gym/compare/v0.6.6...v0.6.7) (2023-06-05)

### [0.6.6](https://github.com/rickstaa/stable-gym/compare/v0.6.3...v0.6.6) (2023-06-05)

### [0.6.5](https://github.com/rickstaa/stable-gym/compare/v0.6.4...v0.6.5) (2023-06-05)

### [0.6.4](https://github.com/rickstaa/stable-gym/compare/v0.6.3...v0.6.4) (2023-06-05)

### [0.6.3](https://github.com/rickstaa/stable-gym/compare/v0.6.2...v0.6.3) (2023-06-05)

### [0.6.2](https://github.com/rickstaa/stable-gym/compare/v0.6.1...v0.6.2) (2023-06-05)

### [0.6.1](https://github.com/rickstaa/stable-gym/compare/v0.6.0...v0.6.1) (2023-06-05)


### Bug Fixes

* fix incorrect stable_gym import ([1a0469d](https://github.com/rickstaa/stable-gym/commit/1a0469d2fbd3def46fc57bbcded07ea39378cfbf))

## [0.6.0](https://github.com/rickstaa/stable-gym/compare/v0.5.18...v0.6.0) (2023-06-05)


### ⚠ BREAKING CHANGES

* The package name is changed so the package should be imported as `stable_gym` in
the future.
* **simzoo:** The package name is changed so the package should be imported as `stable_gym` in
the future.

### build

* **simzoo:** rename `simzoo` package to `stable_gym` ([9db303e](https://github.com/rickstaa/stable-gym/commit/9db303efd2b2256a35d3af48f50ee1a1e1c8ee62))


* Rename simzoo package to stable gym (#117) ([ca28218](https://github.com/rickstaa/stable-gym/commit/ca2821815b1d82abc67f248f0ffdda8bca1634e4)), closes [#117](https://github.com/rickstaa/stable-gym/issues/117)

### [0.5.18](https://github.com/rickstaa/stable-gym/compare/v0.5.16...v0.5.18) (2023-06-05)


### Features

* make simzoo a stand-alone package ([#115](https://github.com/rickstaa/stable-gym/issues/115)) ([942467a](https://github.com/rickstaa/stable-gym/commit/942467aa36f91e438b1a61db69e5278db05ca2c3))


### Bug Fixes

* fix package install bug ([#116](https://github.com/rickstaa/stable-gym/issues/116)) ([f768715](https://github.com/rickstaa/stable-gym/commit/f7687154121c40ec139d60f4a8080a99c6e495ab))

### [0.5.17](https://github.com/rickstaa/stable-gym/compare/v0.5.16...v0.5.17) (2023-06-05)

### [0.5.16](https://github.com/rickstaa/stable-gym/compare/v0.5.15...v0.5.16) (2023-06-05)

### [0.5.15](https://github.com/rickstaa/stable-gym/compare/v0.5.14...v0.5.15) (2023-06-05)

### [0.5.14](https://github.com/rickstaa/stable-gym/compare/v0.5.13...v0.5.14) (2023-06-05)

### [0.5.13](https://github.com/rickstaa/stable-gym/compare/v0.5.12...v0.5.13) (2023-06-02)

### [0.5.12](https://github.com/rickstaa/stable-gym/compare/v0.5.11...v0.5.12) (2023-06-02)


### Features

* increase reference tracking angle threshold ([5f5ca82](https://github.com/rickstaa/stable-gym/commit/5f5ca820f7879f70800343892baa8229ecf702b4))

### [0.5.11](https://github.com/rickstaa/stable-gym/compare/v0.5.10...v0.5.11) (2023-06-02)


### Features

* add reference to CartPoleCost observation ([#108](https://github.com/rickstaa/stable-gym/issues/108)) ([b1f9ea8](https://github.com/rickstaa/stable-gym/commit/b1f9ea8fd69586b57b51e0e7bfaf957af9301cb1))


### Bug Fixes

* change cost function to Han et al. 2020 ([e44293d](https://github.com/rickstaa/stable-gym/commit/e44293d4076a1de180ba268e01e04837092ae006))
* fix 'clip_action' disabled bug and improve docs ([e95cc27](https://github.com/rickstaa/stable-gym/commit/e95cc278cecf98e767e3e74ba12444340a821eff))
* fix a runtime error in the CartPole env ([#110](https://github.com/rickstaa/stable-gym/issues/110)) ([3d9c208](https://github.com/rickstaa/stable-gym/commit/3d9c208330118dabbaf970abb0a4632b2d16d343))
* fix CartPoleCost observation bug ([47699c2](https://github.com/rickstaa/stable-gym/commit/47699c2099f84310513021f3d41eb055cf47a38b))
* fix CartPoleCost observation bug ([37ca1a2](https://github.com/rickstaa/stable-gym/commit/37ca1a20c8c7c68f138992bb7853b9af1f3f42cf))
* fix Oscillator env observation bug ([0634367](https://github.com/rickstaa/stable-gym/commit/06343672980a4f834beb01842f7cf33119f7068c))

### [0.5.10](https://github.com/rickstaa/stable-gym/compare/v0.5.9...v0.5.10) (2023-06-01)

### [0.5.9](https://github.com/rickstaa/stable-gym/compare/v0.5.8...v0.5.9) (2023-06-01)

### [0.5.8](https://github.com/rickstaa/stable-gym/compare/v0.5.7...v0.5.8) (2023-05-31)


### Features

* add reset options argument ([#103](https://github.com/rickstaa/stable-gym/issues/103)) ([a48a873](https://github.com/rickstaa/stable-gym/commit/a48a873902297844515641e97c73c0e3630a27d6))

### [0.5.7](https://github.com/rickstaa/stable-gym/compare/v0.5.6...v0.5.7) (2023-05-31)


### Features

* replace gym with gymnasium ([98cdf0d](https://github.com/rickstaa/stable-gym/commit/98cdf0d7426ed931496e9084787be995d0266f76))
* replace gym with gymnasium ([#100](https://github.com/rickstaa/stable-gym/issues/100)) ([e4f19f3](https://github.com/rickstaa/stable-gym/commit/e4f19f3a426b03a2a36f2c4ed1a069f6fc3a0083))


### Bug Fixes

* fix cartpole rendering ([#99](https://github.com/rickstaa/stable-gym/issues/99)) ([4a97126](https://github.com/rickstaa/stable-gym/commit/4a9712638a0c88be970f6d2f26fa78cd08478dd4))

### [0.5.6](https://github.com/rickstaa/stable-gym/compare/v0.5.5...v0.5.6) (2023-05-30)

### [0.5.5](https://github.com/rickstaa/stable-gym/compare/v0.5.4...v0.5.5) (2023-05-30)

### [0.5.4](https://github.com/rickstaa/stable-gym/compare/v0.5.3...v0.5.4) (2023-05-30)

### [0.5.3](https://github.com/rickstaa/stable-gym/compare/v0.5.2...v0.5.3) (2023-05-30)


### Bug Fixes

* **envs:** fix envs main functions ([32491d5](https://github.com/rickstaa/stable-gym/commit/32491d5e2f265bd80905010c23a7b009d0b1be2a))
* fix cartpole-cost-v0 render mode bug ([fdf04cd](https://github.com/rickstaa/stable-gym/commit/fdf04cd53669a095200d29cae27b8e725e04990f))

### [0.5.2](https://github.com/rickstaa/stable-gym/compare/v0.5.1...v0.5.2) (2023-05-30)


### Features

* **ex3_ekf:** fix function argument bug ([59d4de5](https://github.com/rickstaa/stable-gym/commit/59d4de5afedb434b225dea2220acd8dc365b16d2))

### [0.5.1](https://github.com/rickstaa/stable-gym/compare/v0.5.0...v0.5.1) (2023-05-30)


### Bug Fixes

* **envs:** fix some upstream deprication issues ([d2b2be5](https://github.com/rickstaa/stable-gym/commit/d2b2be5b7084ba346dd0ea7f6d512df8b2316418))

## [0.5.0](https://github.com/rickstaa/stable-gym/compare/v0.4.93...v0.5.0) (2023-05-30)


### ⚠ BREAKING CHANGES

* step now returns 5 values instead of 4.

### Features

* depricate step done return value ([#98](https://github.com/rickstaa/stable-gym/issues/98)) ([4f435d6](https://github.com/rickstaa/stable-gym/commit/4f435d6f9c7aaa12fc91a1722cc5ac4cc22d44a3))

### [0.4.93](https://github.com/rickstaa/stable-gym/compare/v0.4.92...v0.4.93) (2023-05-30)


### Bug Fixes

* **envs:** fix 'render_mode' error ([78c8dfa](https://github.com/rickstaa/stable-gym/commit/78c8dfa93525ddc9daed0c8ac634f156bb3a3277))

### [0.4.92](https://github.com/rickstaa/stable-gym/compare/v0.4.91...v0.4.92) (2023-05-30)

### [0.4.91](https://github.com/rickstaa/stable-gym/compare/v0.4.90...v0.4.91) (2023-05-30)


### Bug Fixes

* fix 'env_spec' not found bug ([#97](https://github.com/rickstaa/stable-gym/issues/97)) ([d5df636](https://github.com/rickstaa/stable-gym/commit/d5df636732c8004e7c9f25c159b9c0d9b9bf61d2))

### [0.4.90](https://github.com/rickstaa/stable-gym/compare/v0.4.89...v0.4.90) (2023-05-30)

### [0.4.89](https://github.com/rickstaa/stable-gym/compare/v0.4.88...v0.4.89) (2023-05-30)

### [0.4.62](https://github.com/rickstaa/stable-gym/compare/v0.4.88...v0.4.62) (2023-05-30)

### [0.4.61](https://github.com/rickstaa/stable-gym/compare/v0.4.88...v0.4.61) (2023-05-30)

### [0.4.53](https://github.com/rickstaa/stable-gym/compare/v0.4.52...v0.4.53) (2022-02-07)

### [0.4.52](https://github.com/rickstaa/stable-gym/compare/v0.4.51...v0.4.52) (2022-02-07)

### [0.4.50](https://github.com/rickstaa/stable-gym/compare/v0.4.49...v0.4.50) (2022-02-03)

### [0.4.51](https://github.com/rickstaa/stable-gym/compare/v0.4.50...v0.4.51) (2022-02-04)

### [0.4.50](https://github.com/rickstaa/stable-gym/compare/v0.4.49...v0.4.50) (2022-02-01)

### [0.4.49](https://github.com/rickstaa/stable-gym/compare/v0.4.48...v0.4.49) (2022-01-26)

### [0.4.46](https://github.com/rickstaa/stable-gym/compare/v0.4.45...v0.4.46) (2022-01-26)

### [0.4.45](https://github.com/rickstaa/stable-gym/compare/v0.4.44...v0.4.45) (2021-12-16)

### [0.4.44](https://github.com/rickstaa/stable-gym/compare/v0.4.43...v0.4.44) (2021-12-13)

### [0.4.43](https://github.com/rickstaa/stable-gym/compare/v0.4.42...v0.4.43) (2021-12-13)

### [0.4.42](https://github.com/rickstaa/stable-gym/compare/v0.4.36...v0.4.42) (2021-12-13)

### [0.4.41](https://github.com/rickstaa/stable-gym/compare/v0.4.40...v0.4.41) (2021-11-22)

### [0.4.40](https://github.com/rickstaa/stable-gym/compare/v0.4.39...v0.4.40) (2021-11-17)

### [0.4.39](https://github.com/rickstaa/stable-gym/compare/v0.4.38...v0.4.39) (2021-11-16)

### [0.4.38](https://github.com/rickstaa/stable-gym/compare/v0.4.37...v0.4.38) (2021-11-14)

### [0.4.37](https://github.com/rickstaa/stable-gym/compare/v0.4.35...v0.4.37) (2021-11-11)

### [0.4.38](https://github.com/rickstaa/stable-gym/compare/v0.4.36...v0.4.38) (2021-12-13)

### [0.4.41](https://github.com/rickstaa/stable-gym/compare/v0.4.40...v0.4.41) (2021-11-22)

### [0.4.40](https://github.com/rickstaa/stable-gym/compare/v0.4.39...v0.4.40) (2021-11-17)

### [0.4.39](https://github.com/rickstaa/stable-gym/compare/v0.4.38...v0.4.39) (2021-11-16)

### [0.4.38](https://github.com/rickstaa/stable-gym/compare/v0.4.37...v0.4.38) (2021-11-14)

### [0.4.37](https://github.com/rickstaa/stable-gym/compare/v0.4.35...v0.4.37) (2021-11-11)

### [0.4.37](https://github.com/rickstaa/stable-gym/compare/v0.4.36...v0.4.37) (2021-12-13)

### [0.4.41](https://github.com/rickstaa/stable-gym/compare/v0.4.40...v0.4.41) (2021-11-22)

### [0.4.40](https://github.com/rickstaa/stable-gym/compare/v0.4.39...v0.4.40) (2021-11-17)

### [0.4.39](https://github.com/rickstaa/stable-gym/compare/v0.4.38...v0.4.39) (2021-11-16)

### [0.4.38](https://github.com/rickstaa/stable-gym/compare/v0.4.37...v0.4.38) (2021-11-14)

### [0.4.37](https://github.com/rickstaa/stable-gym/compare/v0.4.35...v0.4.37) (2021-11-11)

### [0.4.36](https://github.com/rickstaa/stable-gym/compare/v0.4.35...v0.4.36) (2021-12-13)

### [0.4.25](https://github.com/rickstaa/stable-gym/compare/v0.4.21...v0.4.25) (2021-10-07)

### [0.4.24](https://github.com/rickstaa/stable-gym/compare/v0.4.23...v0.4.24) (2021-10-07)

### [0.4.23](https://github.com/rickstaa/stable-gym/compare/v0.4.22...v0.4.23) (2021-10-05)

### [0.4.22](https://github.com/rickstaa/stable-gym/compare/v0.4.20...v0.4.22) (2021-09-28)

### [0.4.21](https://github.com/rickstaa/stable-gym/compare/v0.4.20...v0.4.21) (2021-10-07)

### [0.4.20](https://github.com/rickstaa/stable-gym/compare/v0.4.19...v0.4.20) (2021-09-18)

### [0.4.18](https://github.com/rickstaa/stable-gym/compare/v0.4.17...v0.4.18) (2021-09-13)

### [0.4.17](https://github.com/rickstaa/stable-gym/compare/v0.4.16...v0.4.17) (2021-09-13)

### [0.4.16](https://github.com/rickstaa/stable-gym/compare/v0.4.15...v0.4.16) (2021-09-13)

### [0.4.15](https://github.com/rickstaa/stable-gym/compare/v0.4.14...v0.4.15) (2021-09-13)

### [0.4.14](https://github.com/rickstaa/stable-gym/compare/v0.4.13...v0.4.14) (2021-09-10)

### [0.4.13](https://github.com/rickstaa/stable-gym/compare/v0.4.12...v0.4.13) (2021-09-10)

### [0.4.12](https://github.com/rickstaa/stable-gym/compare/v0.4.11...v0.4.12) (2021-09-10)

### [0.4.11](https://github.com/rickstaa/stable-gym/compare/v0.4.10...v0.4.11) (2021-09-09)

### [0.4.10](https://github.com/rickstaa/stable-gym/compare/v0.4.9...v0.4.10) (2021-09-09)

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Generated by [`auto-changelog`](https://github.com/CookPete/auto-changelog).

## [v0.4.7](https://github.com/rickstaa/stable-gym/compare/v0.4.9...v0.4.7)

## [v0.4.9](https://github.com/rickstaa/stable-gym/compare/v0.4.8...v0.4.9) - 2021-08-21

### Merged

- Update dependency gym to v0.19.0 [`#28`](https://github.com/rickstaa/stable-gym/pull/28)
- Update dependency matplotlib to v3.4.3 [`#27`](https://github.com/rickstaa/stable-gym/pull/27)

## [v0.4.8](https://github.com/rickstaa/stable-gym/compare/v0.4.7...v0.4.8) - 2021-08-21

### Merged

- Update dependency gym to v0.19.0 [`#28`](https://github.com/rickstaa/stable-gym/pull/28)
- Update dependency matplotlib to v3.4.3 [`#27`](https://github.com/rickstaa/stable-gym/pull/27)

### Commits

- :recycle: Fixes some small syntax errors. [`3d30d65`](https://github.com/rickstaa/stable-gym/commit/3d30d65f0f645e0a99649d7676db17062652aa37)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`e670bb2`](https://github.com/rickstaa/stable-gym/commit/e670bb2009e1cb215c68ef9025ed2605095cffae)
- :page_facing_up: Updates changelog [`fed1890`](https://github.com/rickstaa/stable-gym/commit/fed18904985da630714f8436cc9660c9809ce3d0)
- :bookmark: Updates code version to v0.4.7 [`d3452ea`](https://github.com/rickstaa/stable-gym/commit/d3452eab802e9d68d74c8bab1428f99173eae043)

## [v0.4.7](https://github.com/rickstaa/stable-gym/compare/v0.4.6...v0.4.7) - 2021-06-12

### Merged

- :arrow_up: Update dependency gym to v0.18.3 [`#24`](https://github.com/rickstaa/stable-gym/pull/24)
- :arrow_up: Bump gym from 0.18.0 to 0.18.3 [`#25`](https://github.com/rickstaa/stable-gym/pull/25)
- :arrow_up: Update dependency auto-changelog to v2.3.0 [`#26`](https://github.com/rickstaa/stable-gym/pull/26)
- :arrow_up: Bump matplotlib from 3.4.1 to 3.4.2 [`#23`](https://github.com/rickstaa/stable-gym/pull/23)

### Commits

- :sparkles: Adds non-homogenious disturbance [`81f348d`](https://github.com/rickstaa/stable-gym/commit/81f348d5c212b3ffdbd107186037cd31d3347c87)
- :art: Cleans up code structure [`8412d1a`](https://github.com/rickstaa/stable-gym/commit/8412d1a5d23fa818a5eb6b02be6e4f548bec28ba)
- :art: Cleans up code [`3b39e4c`](https://github.com/rickstaa/stable-gym/commit/3b39e4c8bd78914533b042cf1206fcac3060aae0)
- :sparkles: Add disable baseline option to disturber [`3a9928e`](https://github.com/rickstaa/stable-gym/commit/3a9928e237c616242947ed4c525a35b0641a2ac1)
- :sparkles: Adds a recurring impulse to the disturbances [`d009c23`](https://github.com/rickstaa/stable-gym/commit/d009c2321e10341b4a3489f00e30f6fcb2736f7d)
- :memo: Updates changelog [`544d5ea`](https://github.com/rickstaa/stable-gym/commit/544d5eab54ba435e7ded21ba2749accc44eed291)
- :wrench: Switches back to minghoas values for the cartpole [`f92de38`](https://github.com/rickstaa/stable-gym/commit/f92de380f1542b7508b227e7eb46494956821a43)
- :recycle: Simplifies env register code [`1fc3f95`](https://github.com/rickstaa/stable-gym/commit/1fc3f95e45c0f821627f45ddf0a258accae0b26c)
- :art: Fixes some smal syntax errors [`09ed14b`](https://github.com/rickstaa/stable-gym/commit/09ed14b8b6797f62af93e74fd4d1a8831b2a8082)
- :adhesive_bandage: Quickfix for impulse disturbance [`36e6d24`](https://github.com/rickstaa/stable-gym/commit/36e6d24611f550fbd7b582377aca553cc8047a86)
- :bug: Fixes cartpole environment module naming error [`88a5856`](https://github.com/rickstaa/stable-gym/commit/88a585640ab5c4a91c2662f343832325400153ea)
- :bookmark: Bump version: 0.4.6 → 0.4.7 [`ec477d0`](https://github.com/rickstaa/stable-gym/commit/ec477d0a6f0831824bf59de152977f5112b72bc6)
- :bulb: Updates code comments [`c0f3223`](https://github.com/rickstaa/stable-gym/commit/c0f32230f68b7f0353412a848d8b8598cd82d21c)
- :art: Fixes small remark syntax error [`3accf8e`](https://github.com/rickstaa/stable-gym/commit/3accf8e2b9123bccd8fcb1f04ff4cfc99f2130f9)
- :bug: Fixes namespace package name shorting [`6ed66e5`](https://github.com/rickstaa/stable-gym/commit/6ed66e5953be8b8b1731e7bda4cb8ee94a6b48be)
- :twisted_rightwards_arrows: Merge branch 'syntax_fixes' into main [`a1dc2c7`](https://github.com/rickstaa/stable-gym/commit/a1dc2c70ede7dc36e786189fb177fbde58fe8c44)
- :green_heart: Updates gh-action cache [`3a91c29`](https://github.com/rickstaa/stable-gym/commit/3a91c296e660e83d6749a6f4b2748288fdbedb9a)
- :white_check_mark: Fixes bug in python tests [`8dcac68`](https://github.com/rickstaa/stable-gym/commit/8dcac68d91a81d8697d44c2759bd1c1eaef515cf)
- :wrench: Adds missing dependency [`b3b0e18`](https://github.com/rickstaa/stable-gym/commit/b3b0e18df3991c30a75b8259f524dcd887827bec)
- :green_heart: Upates gh-action ubuntu version [`a4a61a4`](https://github.com/rickstaa/stable-gym/commit/a4a61a461ccdb3cb011487cc71a73b33003cbd80)
- :green_heart: Updates gh-action cache [`30af0de`](https://github.com/rickstaa/stable-gym/commit/30af0de0bc7a26667a6a85c49d97d69a23307ccb)
- :rewind: Revert changes made to force the python cache to update [`6d665b7`](https://github.com/rickstaa/stable-gym/commit/6d665b73decd3f2fe862852497c71b8060c1b199)
- :green_heart: Forces python cache to update [`f6d9a8a`](https://github.com/rickstaa/stable-gym/commit/f6d9a8a221c67402919284e084ea425cb517a116)
- :bug: Adds missing dependency [`09627e3`](https://github.com/rickstaa/stable-gym/commit/09627e3d3c00a642722220cbca961d52652f2b82)

## [v0.4.6](https://github.com/rickstaa/stable-gym/compare/v0.4.5...v0.4.6) - 2021-04-21

### Merged

- :sparkles: Add combined disturbance [`#21`](https://github.com/rickstaa/stable-gym/pull/21)
- :rewind: Revert "Add combined disturbance (#19)" [`#20`](https://github.com/rickstaa/stable-gym/pull/20)
- Add combined disturbance [`#19`](https://github.com/rickstaa/stable-gym/pull/19)
- :arrow_up: Update dependency matplotlib to v3.4.1 [`#18`](https://github.com/rickstaa/stable-gym/pull/18)
- :bug: Fixes environment register bug [`#17`](https://github.com/rickstaa/stable-gym/pull/17)
- :sparkles: Add CartPole env [`#15`](https://github.com/rickstaa/stable-gym/pull/15)
- :arrow_up: Update dependency matplotlib to v3.4.0 [`#11`](https://github.com/rickstaa/stable-gym/pull/11)
- :sparkles: Adds pendulum env [`#12`](https://github.com/rickstaa/stable-gym/pull/12)

### Commits

- :fire: Removes environment submodules [`af646d0`](https://github.com/rickstaa/stable-gym/commit/af646d07c2ae6dd2e12a2fc7f53e59060b8a7c99)
- :recycle: Cleans up CartPoleCost environment and adds docs [`7fde58f`](https://github.com/rickstaa/stable-gym/commit/7fde58f608e5a7d0662807f63ff95ad95ccc4c32)
- :fire: Again remove submodules [`b895a4b`](https://github.com/rickstaa/stable-gym/commit/b895a4b875a2144c360a14a2a583a5af031ae9f4)
- :loud_sound: Adds environment information logging [`fc83d36`](https://github.com/rickstaa/stable-gym/commit/fc83d3681200f832b95781e1c00339c25552e772)
- :sparkles: Adds disturber to CartPoleCost environment [`42d7b62`](https://github.com/rickstaa/stable-gym/commit/42d7b62b9a899fcba2520a74117925244a98b52b)
- :fire: Removes old CartPoleCost environment [`53bbcc0`](https://github.com/rickstaa/stable-gym/commit/53bbcc070b9fe7e1cfde4e48e95a48b067a3d0d4)
- :art: This commit cleans up the environments [`7ae5f94`](https://github.com/rickstaa/stable-gym/commit/7ae5f943c7ebe0ef90b0064cb1b30fb2552f7be7)
- :art: Formats the code using the black formatter [`3c64b67`](https://github.com/rickstaa/stable-gym/commit/3c64b67ae209111ce8766afe580ef29ef90e7768)
- :bulb: Updates code comments [`faf3984`](https://github.com/rickstaa/stable-gym/commit/faf39841b6080b820a1f621ca9190251839c0dee)
- :twisted_rightwards_arrows: Merges main branch [`0f9e5cf`](https://github.com/rickstaa/stable-gym/commit/0f9e5cfb0e849e7a8b299c8633a051cfc835d735)
- :bug: Fixes the cartpole cost formula [`42a5dcd`](https://github.com/rickstaa/stable-gym/commit/42a5dcdef02029ea08947b43a10c706a014cdc9e)
- :building_construction: Cleans up the repository structure [`9d48be1`](https://github.com/rickstaa/stable-gym/commit/9d48be11146a8cc5f14f3b5cbf303b6b5f43022e)
- :art: :bug: Fixes some small bugs in the environments [`1d0d3dd`](https://github.com/rickstaa/stable-gym/commit/1d0d3dd883a1484c2c4109b5183f90f4a4ddc951)
- :art: Updates disturbance names [`8c84700`](https://github.com/rickstaa/stable-gym/commit/8c84700b5f254b67d665a09cd2b8d5fbbb8d2cc3)
- :art: Cleans up the codebase [`7b7dbe0`](https://github.com/rickstaa/stable-gym/commit/7b7dbe00d06985f719ada6269ebf5ac12cd53eff)
- :bug: Fixes some small bugs in the disturber [`c465f97`](https://github.com/rickstaa/stable-gym/commit/c465f97b0f73db5f5ab0b8256e42a1aa3e8407bd)
- :art: :bug: Fixes cart_pole_cost env disturber [`99cc58b`](https://github.com/rickstaa/stable-gym/commit/99cc58bb7a86d9e0c9cb3a96f38638d329211c20)
- :bug: Fixes initial disturbance add meganism [`56c680b`](https://github.com/rickstaa/stable-gym/commit/56c680b4fbe2bef5c450d95314ddd31ec7e83049)
- :wrench: Add default disturbance type to the disturber config [`195b332`](https://github.com/rickstaa/stable-gym/commit/195b3324402ff95796077ee8a03409eae6917ab4)
- :mute: Removes environment logging statements [`67a800d`](https://github.com/rickstaa/stable-gym/commit/67a800d3362e3ff5a2383e15a765f779f194c8ea)
- :art: :sparkles: Updates disturbance label precision [`6ba5043`](https://github.com/rickstaa/stable-gym/commit/6ba5043ac14309b61ab9a49c41bb7b4a5364f37c)
- :bug: Fixes env variable range order [`5290928`](https://github.com/rickstaa/stable-gym/commit/5290928fa719fe4d45b55d517906596b2d9ae0b8)
- 🐛Fixes CartPole bugs [`f0c3cfc`](https://github.com/rickstaa/stable-gym/commit/f0c3cfc32b77641bb781188a09f0b1581510c7d0)
- :bug: Removes broken CartPole submodule [`8663436`](https://github.com/rickstaa/stable-gym/commit/8663436ef274ce102562390ec9ade72064ef03fe)
- The environmen is finished. [`2c9e8f1`](https://github.com/rickstaa/stable-gym/commit/2c9e8f12130493e21b00a52f812475ba00107b19)
- Adds dt attribute to CartPole [`98ecb74`](https://github.com/rickstaa/stable-gym/commit/98ecb7407b1b5598987b417458e608fc91d19eed)
- :bug: Fixes CartPoleCost registration bug [`bd60295`](https://github.com/rickstaa/stable-gym/commit/bd60295edd846d7d0d30108ed459f8d26bc7ff66)
- :art: Improves random action sampling [`8f437b7`](https://github.com/rickstaa/stable-gym/commit/8f437b7a4f7eedca411b20ad55138319f2395b6c)
- :bookmark: Bump version: v0.4.5 → 0.4.6 [`de41e51`](https://github.com/rickstaa/stable-gym/commit/de41e51a35f48992f64cdda9dee32734fd93b36f)
- :bug: Fix disturbance value print bug [`c6d0776`](https://github.com/rickstaa/stable-gym/commit/c6d0776bfb11fc789ca92e7a20e7700447ae740d)
- 🐛Fixed CartPole bugs [`632124a`](https://github.com/rickstaa/stable-gym/commit/632124a506c2ccded9929c59d8cb92c69cc84c94)
- :art: Cleans up the environments [`acd28f5`](https://github.com/rickstaa/stable-gym/commit/acd28f51b2e70eab4afe7b526e9b6804b93111fe)
- :truck: Rename CartPoleCost distruber [`b5fa6ec`](https://github.com/rickstaa/stable-gym/commit/b5fa6ec9d4285a2c103a4c6f33bcac2a88c1ee13)
- :wrench: Changes CartPoleCost max_env_steps [`ed47727`](https://github.com/rickstaa/stable-gym/commit/ed477271aab3bfffc9b7dbfff0591809880142a8)
- :art: Improves code structure [`7ea3e45`](https://github.com/rickstaa/stable-gym/commit/7ea3e45c4a219f6640234012603c616467803e6e)
- :bug: Fixes gym environment bug [`8fd2bcc`](https://github.com/rickstaa/stable-gym/commit/8fd2bcc0eb548b1a75f0f98bc2913e424872ec53)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`f8e328a`](https://github.com/rickstaa/stable-gym/commit/f8e328aae83119132d8d487fb0dc28d80bf451c3)
- :memo: Updates CartPoleCost readme.md [`e63f586`](https://github.com/rickstaa/stable-gym/commit/e63f5869d0da737fc8fea5607a8a1243be9ebf5d)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`e180b69`](https://github.com/rickstaa/stable-gym/commit/e180b69da581187af8991756239bb610380136c8)
- :memo: Updates Readme.md [`73d263b`](https://github.com/rickstaa/stable-gym/commit/73d263b547fa2b7304be80a21792efe22813190d)
- :twisted_rightwards_arrows: Merge branch 'dds0117-fixed_cartpole' into main [`c8553ed`](https://github.com/rickstaa/stable-gym/commit/c8553ed97791409fb7a99bcf887bfab6d6bd647e)
- :twisted_rightwards_arrows: Merge branch 'fixed_cartpole' of github.com:dds0117/stable-gym into dds0117-fixed_cartpole [`92b8f8e`](https://github.com/rickstaa/stable-gym/commit/92b8f8e4e6d1dc8b12426ce3219211e7af0149a2)
- :twisted_rightwards_arrows: Merge branch 'dds0117-main' into main [`e9d81d2`](https://github.com/rickstaa/stable-gym/commit/e9d81d261dd2359b6f21faaa312151eb185dc66f)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:dds0117/stable-gym into dds0117-main [`a431e9f`](https://github.com/rickstaa/stable-gym/commit/a431e9f732dd8d3c5e45589db299c7d450edaea5)
- :alien: Updates ex3_ekf submodule [`c6b9ebe`](https://github.com/rickstaa/stable-gym/commit/c6b9ebe1db2eefe34a2ae48ed947cf0f3997a5e8)
- :twisted_rightwards_arrows: Merge branch 'adds_default' into main [`51cad2a`](https://github.com/rickstaa/stable-gym/commit/51cad2a14030bec3c7520321b0f941f727ab43a9)

## [v0.4.5](https://github.com/rickstaa/stable-gym/compare/v0.4.4...v0.4.5) - 2021-03-24

### Commits

- :memo: Updates CHANGELOG.md [`d3fa49d`](https://github.com/rickstaa/stable-gym/commit/d3fa49dbdd62aa85573911fc7320e20f56f48e6f)
- :bookmark: Updates code version to v0.4.5 [`6d51a4a`](https://github.com/rickstaa/stable-gym/commit/6d51a4a14f6d4028d75564e2e8386c6c4554cf70)
- :fire: Removes envs README.md [`8c23119`](https://github.com/rickstaa/stable-gym/commit/8c2311985b8ad14ae26ca935b8bce517b2826e20)

## [v0.4.4](https://github.com/rickstaa/stable-gym/compare/v0.4.3...v0.4.4) - 2021-03-24

### Commits

- :memo: Updates CHANGELOG.md [`0168b5c`](https://github.com/rickstaa/stable-gym/commit/0168b5c9c4484fbc4cb223989b5bbdc9cf257ae1)
- :bookmark: Updates code version to v0.4.4 [`967cdea`](https://github.com/rickstaa/stable-gym/commit/967cdead58b6fa68f4185dbc0f72cf1e01e74b0d)
- :alien: Updates environment submodules [`c07800e`](https://github.com/rickstaa/stable-gym/commit/c07800eea43a5f8fd73883ec7b3cdb6d953ad610)

## [v0.4.3](https://github.com/rickstaa/stable-gym/compare/v0.4.2...v0.4.3) - 2021-03-24

### Commits

- :memo: Updates CHANGELOG.md [`7bd8ca3`](https://github.com/rickstaa/stable-gym/commit/7bd8ca36b03d37b304363cdadcc68680d3177e17)
- :wrench: Updates pytest config [`97c63ae`](https://github.com/rickstaa/stable-gym/commit/97c63ae508381318d71578618fdd807b3e3eec01)
- :bookmark: Updates code version to v0.4.3 [`13af120`](https://github.com/rickstaa/stable-gym/commit/13af12056921614f8e7274c1aa17ed21ecaa763a)

## [v0.4.2](https://github.com/rickstaa/stable-gym/compare/v0.4.1...v0.4.2) - 2021-03-23

### Commits

- :memo: Updates CHANGELOG.md [`9d86f92`](https://github.com/rickstaa/stable-gym/commit/9d86f92939d799961e93f0a8251d39b07e454a48)
- :bookmark: Updates code version to v0.4.2 [`5b8d0af`](https://github.com/rickstaa/stable-gym/commit/5b8d0afe86bc07b0ba8fcf15773c89bfd87a659c)
- :green_heart: Updates stable-gym gh-action [`78378ee`](https://github.com/rickstaa/stable-gym/commit/78378ee05d11d9a96317808c238380662469fc66)

## [v0.4.1](https://github.com/rickstaa/stable-gym/compare/v0.4...v0.4.1) - 2021-03-23

## [v0.4](https://github.com/rickstaa/stable-gym/compare/v0.4.0...v0.4) - 2021-03-24

### Commits

- :bug: Fixes some small bugs in the Disturber class [`2fb3854`](https://github.com/rickstaa/stable-gym/commit/2fb3854213f0c7ad0658448ac25561e959608735)
- :memo: Updates CHANGELOG.md [`0168b5c`](https://github.com/rickstaa/stable-gym/commit/0168b5c9c4484fbc4cb223989b5bbdc9cf257ae1)
- :memo: Updates CHANGELOG.md [`7bd8ca3`](https://github.com/rickstaa/stable-gym/commit/7bd8ca36b03d37b304363cdadcc68680d3177e17)
- :memo: Updates CHANGELOG.md [`9d86f92`](https://github.com/rickstaa/stable-gym/commit/9d86f92939d799961e93f0a8251d39b07e454a48)
- :memo: Updates CHANGELOG.md [`d3fa49d`](https://github.com/rickstaa/stable-gym/commit/d3fa49dbdd62aa85573911fc7320e20f56f48e6f)
- :memo: Updates CHANGELOG.md [`d63ce72`](https://github.com/rickstaa/stable-gym/commit/d63ce720cb4d00d6ad57720b9be6af85e09caf91)
- :wrench: Updates pytest config [`97c63ae`](https://github.com/rickstaa/stable-gym/commit/97c63ae508381318d71578618fdd807b3e3eec01)
- :bookmark: Updates code version to v0.4.5 [`6d51a4a`](https://github.com/rickstaa/stable-gym/commit/6d51a4a14f6d4028d75564e2e8386c6c4554cf70)
- :bookmark: Updates code version to v0.4.4 [`967cdea`](https://github.com/rickstaa/stable-gym/commit/967cdead58b6fa68f4185dbc0f72cf1e01e74b0d)
- :bookmark: Updates code version to v0.4.3 [`13af120`](https://github.com/rickstaa/stable-gym/commit/13af12056921614f8e7274c1aa17ed21ecaa763a)
- :bookmark: Updates code version to v0.4.2 [`5b8d0af`](https://github.com/rickstaa/stable-gym/commit/5b8d0afe86bc07b0ba8fcf15773c89bfd87a659c)
- :bookmark: Updates code version to v0.4.1 [`998aa9b`](https://github.com/rickstaa/stable-gym/commit/998aa9be295a61ba4bae04e2833fd99518cca2c6)
- :fire: Removes envs README.md [`8c23119`](https://github.com/rickstaa/stable-gym/commit/8c2311985b8ad14ae26ca935b8bce517b2826e20)
- :alien: Updates environment submodules [`c07800e`](https://github.com/rickstaa/stable-gym/commit/c07800eea43a5f8fd73883ec7b3cdb6d953ad610)
- :green_heart: Updates stable-gym gh-action [`78378ee`](https://github.com/rickstaa/stable-gym/commit/78378ee05d11d9a96317808c238380662469fc66)

## [v0.4.0](https://github.com/rickstaa/stable-gym/compare/v0.3.7...v0.4.0) - 2021-03-18

### Merged

- :sparkles: Adds a environment disturber class [`#10`](https://github.com/rickstaa/stable-gym/pull/10)

### Commits

- :truck: Changes the name of the parent package [`2808d31`](https://github.com/rickstaa/stable-gym/commit/2808d31c4ce1637a5a5af2d6f8d14141dc701d5c)
- :memo: Updates CHANGELOG.md [`5bf7e03`](https://github.com/rickstaa/stable-gym/commit/5bf7e03342e6a71f0a7a03e674f545526d0714f2)
- :bookmark: Updates code version to v0.4.0 [`46ae85e`](https://github.com/rickstaa/stable-gym/commit/46ae85ec8b521def5862c41f8f83f39f38873d22)
- :alien: Updates versioning of the submodules [`a0f49dc`](https://github.com/rickstaa/stable-gym/commit/a0f49dc4d75e326d00a564c26e7121349fe79e5a)
- :bug: Fixes corrupted submodules [`aee5fb4`](https://github.com/rickstaa/stable-gym/commit/aee5fb4773ccd6a9b57c0b9a88102066738fc19a)

## [v0.3.7](https://github.com/rickstaa/stable-gym/compare/v0.3.6...v0.3.7) - 2021-02-25

### Commits

- :memo: Updates docstrings and submodules [`781684e`](https://github.com/rickstaa/stable-gym/commit/781684e0624d8b94f3e3364b8f9cc70169bcfffc)
- :memo: Updates CHANGELOG.md [`88a6852`](https://github.com/rickstaa/stable-gym/commit/88a685263a14bd921432897317c76f9ba69e72eb)
- :bookmark: Updates code version to v0.3.7 [`6098739`](https://github.com/rickstaa/stable-gym/commit/60987396fd2758000644746e0b4a5f224ce9d37d)

## [v0.3.6](https://github.com/rickstaa/stable-gym/compare/v0.3.5...v0.3.6) - 2021-02-19

### Commits

- :memo: Updates CHANGELOG.md [`1e5f0e4`](https://github.com/rickstaa/stable-gym/commit/1e5f0e43f6de50be5fc77490d3e45008d2adcae9)
- :bookmark: Updates code version to v0.3.6 [`bd36baa`](https://github.com/rickstaa/stable-gym/commit/bd36baa7aaf35f6910f40f6ba6d8f609cc002acb)
- :alien: Updates submodules [`1481a35`](https://github.com/rickstaa/stable-gym/commit/1481a3514903a96502635ac47c18af5487f2f329)

## [v0.3.5](https://github.com/rickstaa/stable-gym/compare/v0.3.4...v0.3.5) - 2021-02-16

### Commits

- :memo: Updates CHANGELOG.md [`888d789`](https://github.com/rickstaa/stable-gym/commit/888d7891b65ba1b18a7f55685dd758f9189ff1f3)
- :bookmark: Updates code version to v0.3.5 [`afb5fd5`](https://github.com/rickstaa/stable-gym/commit/afb5fd5e23c78de7476f7b7fe684823747bef543)
- :bulb: Updates code comments [`3fb1aba`](https://github.com/rickstaa/stable-gym/commit/3fb1aba14c7efa899e16d0090f1c6776c780de12)

## [v0.3.4](https://github.com/rickstaa/stable-gym/compare/v0.3.3...v0.3.4) - 2021-02-04

### Commits

- :bookmark: Updates versioning [`e93eca9`](https://github.com/rickstaa/stable-gym/commit/e93eca97f44a4c1485cd7410c12fc492ab387833)
- :memo: Updates CHANGELOG.md [`ace90af`](https://github.com/rickstaa/stable-gym/commit/ace90afd2c2f466657a8a1f73147071ab775c2ad)
- :bookmark: Updates code version to v0.3.4 [`d317019`](https://github.com/rickstaa/stable-gym/commit/d317019ec264ee63318e2097035569b5518abab7)
- :bookmark: Fixes versioning [`a10010c`](https://github.com/rickstaa/stable-gym/commit/a10010c7145b42b8113f03482defe0d3e5006b80)

## [v0.3.3](https://github.com/rickstaa/stable-gym/compare/v0.3.2...v0.3.3) - 2021-02-04

### Commits

- :alien: Updates submodules [`9518627`](https://github.com/rickstaa/stable-gym/commit/9518627a3976603b5299f6630b334bfe929df331)

## [v0.3.2](https://github.com/rickstaa/stable-gym/compare/v0.3.1...v0.3.2) - 2021-02-04

### Merged

- :arrow_up: Update dependency matplotlib to v3.3.4 [`#9`](https://github.com/rickstaa/stable-gym/pull/9)

### Commits

- :memo: Updates CHANGELOG.md [`2fb1402`](https://github.com/rickstaa/stable-gym/commit/2fb1402d36705bab263ab5e0ee285fa96cd65d9e)
- :bookmark: Updates code version to v0.3.2 [`488b0e2`](https://github.com/rickstaa/stable-gym/commit/488b0e2d5240500ebd557af01bdb8f5511392879)

## [v0.3.1](https://github.com/rickstaa/stable-gym/compare/v0.3...v0.3.1) - 2021-02-04

## [v0.3](https://github.com/rickstaa/stable-gym/compare/v0.3.0...v0.3) - 2021-02-25

### Merged

- :arrow_up: Update dependency matplotlib to v3.3.4 [`#9`](https://github.com/rickstaa/stable-gym/pull/9)

### Commits

- :memo: Updates CHANGELOG.md [`2fb1402`](https://github.com/rickstaa/stable-gym/commit/2fb1402d36705bab263ab5e0ee285fa96cd65d9e)
- :memo: Updates docstrings and submodules [`781684e`](https://github.com/rickstaa/stable-gym/commit/781684e0624d8b94f3e3364b8f9cc70169bcfffc)
- :memo: Updates CHANGELOG.md [`888d789`](https://github.com/rickstaa/stable-gym/commit/888d7891b65ba1b18a7f55685dd758f9189ff1f3)
- :memo: Updates CHANGELOG.md [`422f350`](https://github.com/rickstaa/stable-gym/commit/422f350a6f424a7ebe97fab87352d03b45c294a7)
- :memo: Updates CHANGELOG.md [`1e5f0e4`](https://github.com/rickstaa/stable-gym/commit/1e5f0e43f6de50be5fc77490d3e45008d2adcae9)
- :bookmark: Updates versioning [`e93eca9`](https://github.com/rickstaa/stable-gym/commit/e93eca97f44a4c1485cd7410c12fc492ab387833)
- :memo: Updates CHANGELOG.md [`88a6852`](https://github.com/rickstaa/stable-gym/commit/88a685263a14bd921432897317c76f9ba69e72eb)
- :memo: Updates CHANGELOG.md [`ace90af`](https://github.com/rickstaa/stable-gym/commit/ace90afd2c2f466657a8a1f73147071ab775c2ad)
- :bookmark: Updates code version to v0.3.7 [`6098739`](https://github.com/rickstaa/stable-gym/commit/60987396fd2758000644746e0b4a5f224ce9d37d)
- :bookmark: Updates code version to v0.3.6 [`bd36baa`](https://github.com/rickstaa/stable-gym/commit/bd36baa7aaf35f6910f40f6ba6d8f609cc002acb)
- :bookmark: Updates code version to v0.3.5 [`afb5fd5`](https://github.com/rickstaa/stable-gym/commit/afb5fd5e23c78de7476f7b7fe684823747bef543)
- :bookmark: Updates code version to v0.3.4 [`d317019`](https://github.com/rickstaa/stable-gym/commit/d317019ec264ee63318e2097035569b5518abab7)
- :bookmark: Updates code version to v0.3.2 [`488b0e2`](https://github.com/rickstaa/stable-gym/commit/488b0e2d5240500ebd557af01bdb8f5511392879)
- :bookmark: Updates code version to v0.3.1 [`18574a6`](https://github.com/rickstaa/stable-gym/commit/18574a648d8f1506066700befceaaa4f3820cb09)
- :alien: Updates submodules [`1481a35`](https://github.com/rickstaa/stable-gym/commit/1481a3514903a96502635ac47c18af5487f2f329)
- :alien: Updates submodules [`9518627`](https://github.com/rickstaa/stable-gym/commit/9518627a3976603b5299f6630b334bfe929df331)
- :bulb: Updates code comments [`3fb1aba`](https://github.com/rickstaa/stable-gym/commit/3fb1aba14c7efa899e16d0090f1c6776c780de12)
- :bookmark: Fixes versioning [`a10010c`](https://github.com/rickstaa/stable-gym/commit/a10010c7145b42b8113f03482defe0d3e5006b80)
- :sparkles: Updates oscillator environment [`56d9d65`](https://github.com/rickstaa/stable-gym/commit/56d9d653cc4a035cc35b7c1938e785aa6434d9b2)

## [v0.3.0](https://github.com/rickstaa/stable-gym/compare/v0.2.7...v0.3.0) - 2021-01-18

### Commits

- :memo: Updates CHANGELOG.md [`b653456`](https://github.com/rickstaa/stable-gym/commit/b653456b9a31ef2fe109b598d5f822210a881d29)
- :memo: Updates CHANGELOG.md [`b642769`](https://github.com/rickstaa/stable-gym/commit/b642769a36ae0cf494a9b8bb2df4862542b5136e)
- :memo: Updates CHANGELOG.md [`5b47013`](https://github.com/rickstaa/stable-gym/commit/5b4701359caf46b19bec784454df02d39c58bd05)
- :bookmark: Updates code version to v0.3.0 [`bfe3a6c`](https://github.com/rickstaa/stable-gym/commit/bfe3a6c55dc0d7bd460f4dee15d07108a349079c)
- :bookmark: Updates code version to v0.2.7 [`61f2edf`](https://github.com/rickstaa/stable-gym/commit/61f2edf095dc746d6b5602218fa6095709b0c514)
- :alien: Updates ex3_ekf and oscillator submodules [`e19cd9d`](https://github.com/rickstaa/stable-gym/commit/e19cd9db11ae347b05927a8fd207bbd3b93e024d)

## [v0.2.7](https://github.com/rickstaa/stable-gym/compare/v0.2.6...v0.2.7) - 2021-01-15

### Commits

- :memo: Updates CHANGELOG.md [`6bd099b`](https://github.com/rickstaa/stable-gym/commit/6bd099b7373951634417cc2f747c72980d116f2c)
- :bookmark: Updates code version to v0.2.7 [`ea937a8`](https://github.com/rickstaa/stable-gym/commit/ea937a899471f3a03be745cfe4162a8975598882)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`69fde1c`](https://github.com/rickstaa/stable-gym/commit/69fde1cdb1288e4d2cde3a6b3c7f4c9f25ab313c)
- :memo: Updates README.md [`f143c27`](https://github.com/rickstaa/stable-gym/commit/f143c27496c663d61c01615c1fc340ccf62257f5)

## [v0.2.6](https://github.com/rickstaa/stable-gym/compare/v0.2.5...v0.2.6) - 2021-01-15

### Commits

- :memo: Updates CHANGELOG.md [`8f22c15`](https://github.com/rickstaa/stable-gym/commit/8f22c15cdd26235235be938969ea33c81cabaa6a)
- :bookmark: Updates code version to v0.2.6 [`88829be`](https://github.com/rickstaa/stable-gym/commit/88829becd7debd70ab41b17a32bf4cfe2a75a8b6)
- :green_heart: Removes latest tag [`7b3fa4f`](https://github.com/rickstaa/stable-gym/commit/7b3fa4f758f53695e4df5e8782003deae2549fd4)

## [v0.2.5](https://github.com/rickstaa/stable-gym/compare/v0.2.4...v0.2.5) - 2021-01-15

### Commits

- :memo: Updates CHANGELOG.md [`b6f746b`](https://github.com/rickstaa/stable-gym/commit/b6f746b63512b6431862c42e977ab177a284d45f)
- :wrench: Updates package.json [`2dd4055`](https://github.com/rickstaa/stable-gym/commit/2dd4055ff3747b7776647d2f4f14b1e8b19e5b7f)
- :bookmark: Updates code version to v0.2.5 [`d3f5b4e`](https://github.com/rickstaa/stable-gym/commit/d3f5b4ea660c890db531474e8a0dc10a9f49f5b4)
- :green_heart: Adds latest tag to release gh-action [`374cb1e`](https://github.com/rickstaa/stable-gym/commit/374cb1ed6bc1fea3a6aa279148c15412842a06d4)

## [v0.2.4](https://github.com/rickstaa/stable-gym/compare/v0.2.3...v0.2.4) - 2021-01-15

### Merged

- :pushpin: Pin dependency auto-changelog to 2.2.1 [`#8`](https://github.com/rickstaa/stable-gym/pull/8)
- :green_heart: Adds release gh-action [`#7`](https://github.com/rickstaa/stable-gym/pull/7)
- :arrow_up: Update reviewdog/action-flake8 action to v3 [`#4`](https://github.com/rickstaa/stable-gym/pull/4)
- :arrow_up: Update reviewdog/action-black action to v2 [`#5`](https://github.com/rickstaa/stable-gym/pull/5)
- :arrow_up: Update reviewdog/action-remark-lint action to v2 [`#6`](https://github.com/rickstaa/stable-gym/pull/6)

### Commits

- :arrow_up: Updates submodules [`f75d735`](https://github.com/rickstaa/stable-gym/commit/f75d7350597958d552f159d959cf13038a8fa816)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`04f2b85`](https://github.com/rickstaa/stable-gym/commit/04f2b85101cd71bc4f47087513781233f75086f7)
- :twisted_rightwards_arrows: Merge branch 'adds_release_gh_action' into main [`c54ffab`](https://github.com/rickstaa/stable-gym/commit/c54ffab1bd54e8f198fb625fb9cb4b013daba567)
- :green_heart: Updates release gh-action and submodules [`ff33fea`](https://github.com/rickstaa/stable-gym/commit/ff33feacf568a94833857eda51fee5571b2b9bd9)
- :green_heart: Updates gh-actions and submodules [`ca364df`](https://github.com/rickstaa/stable-gym/commit/ca364df1a6a3cbd96283bf5061c9b148b40a9142)
- :green_heart: Fixes gh-action names [`a0da4fe`](https://github.com/rickstaa/stable-gym/commit/a0da4fe249d73bc526bbd9531dba960127cfc571)

## [v0.2.3](https://github.com/rickstaa/stable-gym/compare/v0.2.2...v0.2.3) - 2021-01-12

### Commits

- :green_heart: Adds release gh-action [`b0b7b1d`](https://github.com/rickstaa/stable-gym/commit/b0b7b1d456b53a4cc2fa295121db4aa0452bce5c)
- :green_heart: Fixes gh-actions and updates submodules [`dec68f8`](https://github.com/rickstaa/stable-gym/commit/dec68f8825edcd487deb1977ace41f48b22679a4)
- :recycle: Cleans up code [`4ed3c94`](https://github.com/rickstaa/stable-gym/commit/4ed3c948b9b417252a5747a410640617471203fb)
- :green_heart: Fixes release gh-action and updates submodules [`4ccd90e`](https://github.com/rickstaa/stable-gym/commit/4ccd90eefe8f45addbc2a96311dda17eb83fc161)
- :memo: Updates changelog [`b92c8f0`](https://github.com/rickstaa/stable-gym/commit/b92c8f0b25ec4334e72a63b946f609de74df45ae)
- :green_heart: Updates gh-action [`e21b53f`](https://github.com/rickstaa/stable-gym/commit/e21b53f743c0bd8a6095b2cb9ec06090c31fc8b4)

## [v0.2.2](https://github.com/rickstaa/stable-gym/compare/v0.2.1...v0.2.2) - 2020-12-19

### Commits

- :memo: Updates changelog [`0d3edb0`](https://github.com/rickstaa/stable-gym/commit/0d3edb03f342c497f39c958202e945dba70bc92a)
- :bookmark: Bump version: 0.2.1 → 0.2.2 [`7c043dc`](https://github.com/rickstaa/stable-gym/commit/7c043dc5085bcabea3cedd1b3a2031e06344e48e)
- :white_check_mark: Updates tests [`1aea7fb`](https://github.com/rickstaa/stable-gym/commit/1aea7fbd24196366be4fdc8a215415add9287ece)

## [v0.2.1](https://github.com/rickstaa/stable-gym/compare/v0.2...v0.2.1) - 2020-12-19

## [v0.2](https://github.com/rickstaa/stable-gym/compare/v0.2.0...v0.2) - 2021-01-15

### Merged

- :pushpin: Pin dependency auto-changelog to 2.2.1 [`#8`](https://github.com/rickstaa/stable-gym/pull/8)
- :green_heart: Adds release gh-action [`#7`](https://github.com/rickstaa/stable-gym/pull/7)
- :arrow_up: Update reviewdog/action-flake8 action to v3 [`#4`](https://github.com/rickstaa/stable-gym/pull/4)
- :arrow_up: Update reviewdog/action-black action to v2 [`#5`](https://github.com/rickstaa/stable-gym/pull/5)
- :arrow_up: Update reviewdog/action-remark-lint action to v2 [`#6`](https://github.com/rickstaa/stable-gym/pull/6)

### Commits

- :memo: Updates CHANGELOG.md [`b6f746b`](https://github.com/rickstaa/stable-gym/commit/b6f746b63512b6431862c42e977ab177a284d45f)
- :green_heart: Fixes gh-actions and updates submodules [`dec68f8`](https://github.com/rickstaa/stable-gym/commit/dec68f8825edcd487deb1977ace41f48b22679a4)
- :construction_worker: Adds github linting and release actions [`7c5a1dc`](https://github.com/rickstaa/stable-gym/commit/7c5a1dcf5396a32606c877cbbfe74c2d3bd5b874)
- :memo: Updates changelog [`22c1eee`](https://github.com/rickstaa/stable-gym/commit/22c1eee1750975bf130539aa6b5509aa0c16586e)
- :wrench: Updates package.json [`2dd4055`](https://github.com/rickstaa/stable-gym/commit/2dd4055ff3747b7776647d2f4f14b1e8b19e5b7f)
- :recycle: Cleans up code [`4ed3c94`](https://github.com/rickstaa/stable-gym/commit/4ed3c948b9b417252a5747a410640617471203fb)
- :memo: Updates CHANGELOG.md [`6bd099b`](https://github.com/rickstaa/stable-gym/commit/6bd099b7373951634417cc2f747c72980d116f2c)
- :green_heart: Updates release gh-action and submodules [`ff33fea`](https://github.com/rickstaa/stable-gym/commit/ff33feacf568a94833857eda51fee5571b2b9bd9)
- :green_heart: Fixes release gh-action and updates submodules [`4ccd90e`](https://github.com/rickstaa/stable-gym/commit/4ccd90eefe8f45addbc2a96311dda17eb83fc161)
- :bookmark: Updates code version to v0.2.7 [`ea937a8`](https://github.com/rickstaa/stable-gym/commit/ea937a899471f3a03be745cfe4162a8975598882)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`69fde1c`](https://github.com/rickstaa/stable-gym/commit/69fde1cdb1288e4d2cde3a6b3c7f4c9f25ab313c)
- :memo: Updates CHANGELOG.md [`8f22c15`](https://github.com/rickstaa/stable-gym/commit/8f22c15cdd26235235be938969ea33c81cabaa6a)
- :memo: Updates changelog [`b92c8f0`](https://github.com/rickstaa/stable-gym/commit/b92c8f0b25ec4334e72a63b946f609de74df45ae)
- :memo: Updates changelog [`0d3edb0`](https://github.com/rickstaa/stable-gym/commit/0d3edb03f342c497f39c958202e945dba70bc92a)
- :bookmark: Updates code version to v0.2.6 [`88829be`](https://github.com/rickstaa/stable-gym/commit/88829becd7debd70ab41b17a32bf4cfe2a75a8b6)
- :bookmark: Updates code version to v0.2.5 [`d3f5b4e`](https://github.com/rickstaa/stable-gym/commit/d3f5b4ea660c890db531474e8a0dc10a9f49f5b4)
- :green_heart: Updates gh-actions and submodules [`ca364df`](https://github.com/rickstaa/stable-gym/commit/ca364df1a6a3cbd96283bf5061c9b148b40a9142)
- :bookmark: Bump version: 0.2.1 → 0.2.2 [`7c043dc`](https://github.com/rickstaa/stable-gym/commit/7c043dc5085bcabea3cedd1b3a2031e06344e48e)
- :bookmark: Bump version: 0.2.0 → 0.2.1 [`e855b4e`](https://github.com/rickstaa/stable-gym/commit/e855b4e6cabaa1f13467e9682f066ab543406046)
- :bookmark: Fixes version tags [`47fa195`](https://github.com/rickstaa/stable-gym/commit/47fa195d2f3e0beb4ebb0a83344fa34e60b2cc43)
- :green_heart: Removes latest tag [`7b3fa4f`](https://github.com/rickstaa/stable-gym/commit/7b3fa4f758f53695e4df5e8782003deae2549fd4)
- :green_heart: Adds latest tag to release gh-action [`374cb1e`](https://github.com/rickstaa/stable-gym/commit/374cb1ed6bc1fea3a6aa279148c15412842a06d4)
- :white_check_mark: Updates tests [`1aea7fb`](https://github.com/rickstaa/stable-gym/commit/1aea7fbd24196366be4fdc8a215415add9287ece)
- :bug: Fixes missing dependencies for the github actions [`225b8b2`](https://github.com/rickstaa/stable-gym/commit/225b8b2a93f1e57e8b92b5cdd70131af6c033f15)
- :arrow_up: Updates submodules [`f75d735`](https://github.com/rickstaa/stable-gym/commit/f75d7350597958d552f159d959cf13038a8fa816)
- :memo: Updates README.md [`f143c27`](https://github.com/rickstaa/stable-gym/commit/f143c27496c663d61c01615c1fc340ccf62257f5)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`04f2b85`](https://github.com/rickstaa/stable-gym/commit/04f2b85101cd71bc4f47087513781233f75086f7)
- :twisted_rightwards_arrows: Merge branch 'adds_release_gh_action' into main [`c54ffab`](https://github.com/rickstaa/stable-gym/commit/c54ffab1bd54e8f198fb625fb9cb4b013daba567)
- :art: Fixes some syntax errors in the submodules [`4e1bc80`](https://github.com/rickstaa/stable-gym/commit/4e1bc80ceeefea99ba6395ca2d28e54148308af7)
- :green_heart: Fixes gh-action names [`a0da4fe`](https://github.com/rickstaa/stable-gym/commit/a0da4fe249d73bc526bbd9531dba960127cfc571)
- :green_heart: Updates gh-action [`e21b53f`](https://github.com/rickstaa/stable-gym/commit/e21b53f743c0bd8a6095b2cb9ec06090c31fc8b4)

## [v0.2.0](https://github.com/rickstaa/stable-gym/compare/v0.1.9...v0.2.0) - 2020-12-19

### Commits

- :memo: Updates changelog [`aa8b060`](https://github.com/rickstaa/stable-gym/commit/aa8b060f5ed9f93eaba5702c4efa28986f9fcbf3)
- :alien: Updates submodules [`f3f91fe`](https://github.com/rickstaa/stable-gym/commit/f3f91fef400fb5ed214695f5085b581c980bf15f)
- :arrow_up: Merge pull request #3 from rickstaa/renovate/matplotlib-3.x [`5385ab3`](https://github.com/rickstaa/stable-gym/commit/5385ab3a00db941cb24eedc24d57b5057c2a070d)
- Update dependency matplotlib to v3.3.3 [`310135d`](https://github.com/rickstaa/stable-gym/commit/310135dabd47292212746b3e08df0b50da29c024)
- :arrow_up: Merge pull request #2 from rickstaa/renovate/gym-0.x [`aa4ba0f`](https://github.com/rickstaa/stable-gym/commit/aa4ba0fe07508c2bdf4b7d45c6ca416b2b60b9f2)
- Update dependency gym to v0.18.0 [`105c4ae`](https://github.com/rickstaa/stable-gym/commit/105c4ae1548b7500121f1095739ac14f93c71ff8)
- :construction_worker: Merge pull request #1 from rickstaa/renovate/configure [`2a7e126`](https://github.com/rickstaa/stable-gym/commit/2a7e126da2e626f9340aad2b5afcc07daf3a35c7)
- :memo: Updates changelog [`23d2fc6`](https://github.com/rickstaa/stable-gym/commit/23d2fc6cebc05b0844ad7377b53326d5f453e5a6)
- Add renovate.json [`909aa24`](https://github.com/rickstaa/stable-gym/commit/909aa24f5fcf9d024a62c5097ba5298c72ea69b8)

## [v0.1.9](https://github.com/rickstaa/stable-gym/compare/v0.1.8...v0.1.9) - 2020-12-15

### Commits

- :memo: Updates changelog [`38b9105`](https://github.com/rickstaa/stable-gym/commit/38b91055adeefc660eb9401263968fee0442d18a)
- :bookmark: Bump version: 0.1.8 → 0.1.9 [`87fe1e1`](https://github.com/rickstaa/stable-gym/commit/87fe1e16c73daae8ea264803cfce587ef240edb3)
- :up_arrow: Updates submodules [`1379584`](https://github.com/rickstaa/stable-gym/commit/13795848cabb263f521074bff6262cbbc8ac753f)

## [v0.1.8](https://github.com/rickstaa/stable-gym/compare/v0.1.7...v0.1.8) - 2020-12-15

### Commits

- :memo: Adds contributing guidelines [`80bda36`](https://github.com/rickstaa/stable-gym/commit/80bda3664bd3ac4f09ceba19b98032395c379325)
- :memo: Updates changelog [`ce26868`](https://github.com/rickstaa/stable-gym/commit/ce26868356848e63e9e9e937e0b88802208fcb29)
- :bookmark: Bump version: 0.1.7 → 0.1.8 [`759504d`](https://github.com/rickstaa/stable-gym/commit/759504d69cdcd5b9a6fbd71edcc72bdd32bc0871)

## [v0.1.7](https://github.com/rickstaa/stable-gym/compare/v0.1.6...v0.1.7) - 2020-12-15

### Commits

- :up_arrow: Updates submodules and fixes bumpversion bug [`8aca03b`](https://github.com/rickstaa/stable-gym/commit/8aca03b388300b604e91c386f6bd0bf6a19947a7)
- :memo: Updates changelog [`eb62904`](https://github.com/rickstaa/stable-gym/commit/eb629047ba6bee04b367c9f0af3bbd96de9ea072)
- :bookmark: Bump version: 0.1.6 → 0.1.7 [`b2f821e`](https://github.com/rickstaa/stable-gym/commit/b2f821efb2a92dc62cec69416654df9831fceea6)
- :wrench: Adds setup.cfg to bumpversion files [`f2c55ca`](https://github.com/rickstaa/stable-gym/commit/f2c55ca0ea17711eaf36649afd513344e1571e23)

## [v0.1.6](https://github.com/rickstaa/stable-gym/compare/v0.1.5...v0.1.6) - 2020-12-15

### Commits

- :memo: Updates changelog [`5de6eeb`](https://github.com/rickstaa/stable-gym/commit/5de6eeba8481eae051836e287f8461a6f24fcfa4)
- :arrow_up: Updates submodules environments [`ed608a3`](https://github.com/rickstaa/stable-gym/commit/ed608a34be90ae95e866a4231ba313b0ad43b17a)
- :art: Add __version__ variable [`3e2dbed`](https://github.com/rickstaa/stable-gym/commit/3e2dbed2b737e685cc56fd0851d66af990e5f7f7)

## [v0.1.5](https://github.com/rickstaa/stable-gym/compare/v0.1.4...v0.1.5) - 2020-12-07

### Commits

- :memo: Updates changelog [`f4cf86e`](https://github.com/rickstaa/stable-gym/commit/f4cf86e9710947da0506c5501a3741dbc1b2dacc)
- :memo: Updates CHANGELOG.md [`7e3ba63`](https://github.com/rickstaa/stable-gym/commit/7e3ba6342480976138570438849f626507bf4738)
- :art: Improves code formatting [`3a8b688`](https://github.com/rickstaa/stable-gym/commit/3a8b688bae33c31f8766d1d430be55f7674540c5)
- :bookmark: Bump version: 0.1.4 → 0.1.5 [`b6c8a51`](https://github.com/rickstaa/stable-gym/commit/b6c8a51bc41c76cf81f00f676e1d5d2d4b1e4472)

## [v0.1.4](https://github.com/rickstaa/stable-gym/compare/v0.1.3...v0.1.4) - 2020-12-04

### Commits

- :memo: Updates CHANGELOG.md [`3fca29c`](https://github.com/rickstaa/stable-gym/commit/3fca29c1a7ac596b674cc7e8a5c8708be546e711)
- :wrench: Updates pyproject.toml file [`c1ba089`](https://github.com/rickstaa/stable-gym/commit/c1ba0899435c1c6208e86fb5a9f5017729438b84)
- :bookmark: Bump version: 0.1.3 → 0.1.4 [`6e4917e`](https://github.com/rickstaa/stable-gym/commit/6e4917ef9b37e9eded87126fd7b171c9ce544edc)

## [v0.1.3](https://github.com/rickstaa/stable-gym/compare/v0.1.2...v0.1.3) - 2020-12-04

### Commits

- :bookmark: Bump version: 0.1.2 → 0.1.3 [`5a4261d`](https://github.com/rickstaa/stable-gym/commit/5a4261dc2e5f2e178e73fab40109ac6c59d42bd7)
- :wrench: Updates pip package url [`3564e12`](https://github.com/rickstaa/stable-gym/commit/3564e120cb44a87ed8f329da0a63ad92c63a4de2)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`029e2bf`](https://github.com/rickstaa/stable-gym/commit/029e2bfc4d48f8817ef2a3714e5a827ce7e79c65)
- :memo: Updates CHANGELOG [`252532c`](https://github.com/rickstaa/stable-gym/commit/252532c61070da91d5fff0e9e0ff646dfd507174)
- page_facing_up: Updates CHANGELOG [`9019d70`](https://github.com/rickstaa/stable-gym/commit/9019d703ba969a84347bfe908662f775f26906f2)

## [v0.1.2](https://github.com/rickstaa/stable-gym/compare/v0.1.1...v0.1.2) - 2020-12-03

### Commits

- :bookmark: Bump version: 0.1.1 → 0.1.2 [`631f76d`](https://github.com/rickstaa/stable-gym/commit/631f76dfc7065f3c2d7cf6a79df724c63ecc22c9)
- :alien: Updates environment submodules [`ec58e44`](https://github.com/rickstaa/stable-gym/commit/ec58e440411700ce0cefefc6fcc790b1ad310216)

## [v0.1.1](https://github.com/rickstaa/stable-gym/compare/v0.1.0...v0.1.1) - 2020-12-03

### Commits

- :bookmark: Bump version: 0.1.0 → 0.1.1 [`5f7f42c`](https://github.com/rickstaa/stable-gym/commit/5f7f42cbe7826e7ed93dfedaa0ddb0e90f7738db)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`77f2024`](https://github.com/rickstaa/stable-gym/commit/77f20249f1a4a15be6ea2efd95dbb6ded71dcc72)
- :memo: Updates README.md [`e6b96ab`](https://github.com/rickstaa/stable-gym/commit/e6b96ab8c4a638320b273860626e6d4641cd50b4)
- :memo: Updates README.md [`1656e45`](https://github.com/rickstaa/stable-gym/commit/1656e45347fdca00536786dbbcda50fdb104fe5f)

## [v0.1.0](https://github.com/rickstaa/stable-gym/compare/v0.0.3...v0.1.0) - 2020-12-03

### Commits

- :sparkles: Adds Ex3_EKF environment git submodule [`f6ac9ff`](https://github.com/rickstaa/stable-gym/commit/f6ac9ff968e4ec527eb1d7e4ca357b4e43a33408)
- :memo: Updates README.md [`38a4dc4`](https://github.com/rickstaa/stable-gym/commit/38a4dc4edaa87effc950233f05bd4c7d365f598f)
- :alien: Updates Oscillator environment version [`9450234`](https://github.com/rickstaa/stable-gym/commit/9450234c4971b1592c84d7ba75681d49306aacbc)
- :alien: Updates environment submodules [`e14184c`](https://github.com/rickstaa/stable-gym/commit/e14184c5fa3f1cfdbadb87c3d445dc02b60b76aa)
- :white_check_mark: Updates tests [`99f7f03`](https://github.com/rickstaa/stable-gym/commit/99f7f03c7f98b4619b570108954ecc7535e584bc)
- :bug: Fixes bug in the pyproject.toml [`ee268ae`](https://github.com/rickstaa/stable-gym/commit/ee268aeb7e97770215bc842a099e80d42bef5191)
- :bookmark: Bump version: 0.0.3 → 0.1.0 [`747ed41`](https://github.com/rickstaa/stable-gym/commit/747ed41b603e76798083ca27485462092d36e418)
- :alien: Updates environment submodules [`ca1ead1`](https://github.com/rickstaa/stable-gym/commit/ca1ead19cd8b05d92f6981090151e11fa7e1b2c1)
- :wrench: Updates bumpversion configuration file [`2423841`](https://github.com/rickstaa/stable-gym/commit/24238418f3f25917900b1cab4ea24c89b3a9f5cd)

## [v0.0.3](https://github.com/rickstaa/stable-gym/compare/v0.0.2...v0.0.3) - 2020-12-03

### Commits

- :sparkles: Updates oscillator environment [`696fa9d`](https://github.com/rickstaa/stable-gym/commit/696fa9d5c8c2ad85c6433cb38f1e7fdfe310c522)
- :package: Changes package to namespace package [`2f13f1e`](https://github.com/rickstaa/stable-gym/commit/2f13f1e01d68f4a3e34a549ddbc88291b0fe247a)
- :sparkles: Updates oscillator submodule [`ce19923`](https://github.com/rickstaa/stable-gym/commit/ce1992340c540a694802595f80ccca1ccbc774bb)
- :arrow_up: Updates setup.py file and environment registration method [`55b5128`](https://github.com/rickstaa/stable-gym/commit/55b51288572d7648e398131c68f7727077f6833f)
- :wrench: Updates setuptools cfg file [`7386a49`](https://github.com/rickstaa/stable-gym/commit/7386a49a1fc3919df9c2071cb30f224e25a12927)
- :pencil: Updates README [`4281343`](https://github.com/rickstaa/stable-gym/commit/42813436f6045a0fd36acb06e50130e126e7bd39)
- :heavy_minus_sign: Depricate setuptools_scm and use bump2version instead [`1ee46ce`](https://github.com/rickstaa/stable-gym/commit/1ee46ce732284d2aaa04e9e63badecb61597b981)
- :bookmark: Bump version: 0.0.2 → 0.0.3 [`9288f7f`](https://github.com/rickstaa/stable-gym/commit/9288f7ff96e50dc1daafbc8b2d1e36695a01ac3e)
- :sparkles: Updates oscillator environment [`635184a`](https://github.com/rickstaa/stable-gym/commit/635184add4444696b18bf8f4ceb3bdec90fa61b0)
- :bug: Updates oscillator submodule [`87296d0`](https://github.com/rickstaa/stable-gym/commit/87296d0cd0b0cdfa656bf01631e9c3c885b0e88d)
- :bug: Updates oscillator module [`96f3b6d`](https://github.com/rickstaa/stable-gym/commit/96f3b6d44397a390618aab630ff9ee7060ff2264)
- :fire: Removes unused files [`1f6ae24`](https://github.com/rickstaa/stable-gym/commit/1f6ae243149e6c4834733af3ac9e333c22968bd5)

## [v0.0.2](https://github.com/rickstaa/stable-gym/compare/v0.0.1...v0.0.2) - 2020-08-14

### Commits

- :pencil: Updates readme [`66596c1`](https://github.com/rickstaa/stable-gym/commit/66596c17a2b882b8b6c9f86e2fdb553291d6eb62)
- :sparkles: Updates oscillator submodule [`953982e`](https://github.com/rickstaa/stable-gym/commit/953982ed5b519ef8dcb7679722603029fdcb393a)

## [v0.0.1](https://github.com/rickstaa/stable-gym/compare/v0...v0.0.1) - 2020-08-14

## [v0](https://github.com/rickstaa/stable-gym/compare/v0.0.0...v0) - 2021-03-24

### Merged

- :sparkles: Adds a environment disturber class [`#10`](https://github.com/rickstaa/stable-gym/pull/10)
- :arrow_up: Update dependency matplotlib to v3.3.4 [`#9`](https://github.com/rickstaa/stable-gym/pull/9)
- :pushpin: Pin dependency auto-changelog to 2.2.1 [`#8`](https://github.com/rickstaa/stable-gym/pull/8)
- :green_heart: Adds release gh-action [`#7`](https://github.com/rickstaa/stable-gym/pull/7)
- :arrow_up: Update reviewdog/action-flake8 action to v3 [`#4`](https://github.com/rickstaa/stable-gym/pull/4)
- :arrow_up: Update reviewdog/action-black action to v2 [`#5`](https://github.com/rickstaa/stable-gym/pull/5)
- :arrow_up: Update reviewdog/action-remark-lint action to v2 [`#6`](https://github.com/rickstaa/stable-gym/pull/6)

### Commits

- :bug: Fixes some small bugs in the Disturber class [`2fb3854`](https://github.com/rickstaa/stable-gym/commit/2fb3854213f0c7ad0658448ac25561e959608735)
- :memo: Updates CHANGELOG.md [`b6f746b`](https://github.com/rickstaa/stable-gym/commit/b6f746b63512b6431862c42e977ab177a284d45f)
- :green_heart: Fixes gh-actions and updates submodules [`dec68f8`](https://github.com/rickstaa/stable-gym/commit/dec68f8825edcd487deb1977ace41f48b22679a4)
- :construction_worker: Adds github linting and release actions [`7c5a1dc`](https://github.com/rickstaa/stable-gym/commit/7c5a1dcf5396a32606c877cbbfe74c2d3bd5b874)
- :truck: Changes the name of the parent package [`2808d31`](https://github.com/rickstaa/stable-gym/commit/2808d31c4ce1637a5a5af2d6f8d14141dc701d5c)
- :memo: Updates changelog [`22c1eee`](https://github.com/rickstaa/stable-gym/commit/22c1eee1750975bf130539aa6b5509aa0c16586e)
- :memo: Adds contributing guidelines [`80bda36`](https://github.com/rickstaa/stable-gym/commit/80bda3664bd3ac4f09ceba19b98032395c379325)
- :memo: Updates CHANGELOG.md [`0168b5c`](https://github.com/rickstaa/stable-gym/commit/0168b5c9c4484fbc4cb223989b5bbdc9cf257ae1)
- :memo: Updates CHANGELOG.md [`b653456`](https://github.com/rickstaa/stable-gym/commit/b653456b9a31ef2fe109b598d5f822210a881d29)
- :wrench: Updates package.json [`2dd4055`](https://github.com/rickstaa/stable-gym/commit/2dd4055ff3747b7776647d2f4f14b1e8b19e5b7f)
- :memo: Updates CHANGELOG.md [`7bd8ca3`](https://github.com/rickstaa/stable-gym/commit/7bd8ca36b03d37b304363cdadcc68680d3177e17)
- :memo: Updates changelog [`f4cf86e`](https://github.com/rickstaa/stable-gym/commit/f4cf86e9710947da0506c5501a3741dbc1b2dacc)
- :recycle: Cleans up code [`4ed3c94`](https://github.com/rickstaa/stable-gym/commit/4ed3c948b9b417252a5747a410640617471203fb)
- :memo: Updates CHANGELOG.md [`9d86f92`](https://github.com/rickstaa/stable-gym/commit/9d86f92939d799961e93f0a8251d39b07e454a48)
- :memo: Updates CHANGELOG.md [`b642769`](https://github.com/rickstaa/stable-gym/commit/b642769a36ae0cf494a9b8bb2df4862542b5136e)
- :memo: Updates CHANGELOG.md [`d3fa49d`](https://github.com/rickstaa/stable-gym/commit/d3fa49dbdd62aa85573911fc7320e20f56f48e6f)
- :memo: Updates changelog [`aa8b060`](https://github.com/rickstaa/stable-gym/commit/aa8b060f5ed9f93eaba5702c4efa28986f9fcbf3)
- :memo: Updates CHANGELOG.md [`3fca29c`](https://github.com/rickstaa/stable-gym/commit/3fca29c1a7ac596b674cc7e8a5c8708be546e711)
- :memo: Updates CHANGELOG.md [`2fb1402`](https://github.com/rickstaa/stable-gym/commit/2fb1402d36705bab263ab5e0ee285fa96cd65d9e)
- :memo: Updates docstrings and submodules [`781684e`](https://github.com/rickstaa/stable-gym/commit/781684e0624d8b94f3e3364b8f9cc70169bcfffc)
- :memo: Updates CHANGELOG.md [`888d789`](https://github.com/rickstaa/stable-gym/commit/888d7891b65ba1b18a7f55685dd758f9189ff1f3)
- :memo: Updates CHANGELOG.md [`422f350`](https://github.com/rickstaa/stable-gym/commit/422f350a6f424a7ebe97fab87352d03b45c294a7)
- :memo: Updates CHANGELOG.md [`5b47013`](https://github.com/rickstaa/stable-gym/commit/5b4701359caf46b19bec784454df02d39c58bd05)
- :memo: Updates CHANGELOG.md [`1e5f0e4`](https://github.com/rickstaa/stable-gym/commit/1e5f0e43f6de50be5fc77490d3e45008d2adcae9)
- :bookmark: Updates versioning [`e93eca9`](https://github.com/rickstaa/stable-gym/commit/e93eca97f44a4c1485cd7410c12fc492ab387833)
- :green_heart: Updates release gh-action and submodules [`ff33fea`](https://github.com/rickstaa/stable-gym/commit/ff33feacf568a94833857eda51fee5571b2b9bd9)
- :memo: Updates CHANGELOG.md [`5bf7e03`](https://github.com/rickstaa/stable-gym/commit/5bf7e03342e6a71f0a7a03e674f545526d0714f2)
- :memo: Updates CHANGELOG.md [`88a6852`](https://github.com/rickstaa/stable-gym/commit/88a685263a14bd921432897317c76f9ba69e72eb)
- :memo: Updates CHANGELOG.md [`d63ce72`](https://github.com/rickstaa/stable-gym/commit/d63ce720cb4d00d6ad57720b9be6af85e09caf91)
- :green_heart: Fixes release gh-action and updates submodules [`4ccd90e`](https://github.com/rickstaa/stable-gym/commit/4ccd90eefe8f45addbc2a96311dda17eb83fc161)
- :memo: Updates changelog [`b92c8f0`](https://github.com/rickstaa/stable-gym/commit/b92c8f0b25ec4334e72a63b946f609de74df45ae)
- :memo: Updates changelog [`0d3edb0`](https://github.com/rickstaa/stable-gym/commit/0d3edb03f342c497f39c958202e945dba70bc92a)
- :wrench: Updates pytest config [`97c63ae`](https://github.com/rickstaa/stable-gym/commit/97c63ae508381318d71578618fdd807b3e3eec01)
- :up_arrow: Updates submodules and fixes bumpversion bug [`8aca03b`](https://github.com/rickstaa/stable-gym/commit/8aca03b388300b604e91c386f6bd0bf6a19947a7)
- :memo: Updates CHANGELOG.md [`ace90af`](https://github.com/rickstaa/stable-gym/commit/ace90afd2c2f466657a8a1f73147071ab775c2ad)
- :bookmark: Updates code version to v0.4.5 [`6d51a4a`](https://github.com/rickstaa/stable-gym/commit/6d51a4a14f6d4028d75564e2e8386c6c4554cf70)
- :bookmark: Updates code version to v0.4.4 [`967cdea`](https://github.com/rickstaa/stable-gym/commit/967cdead58b6fa68f4185dbc0f72cf1e01e74b0d)
- :bookmark: Updates code version to v0.4.3 [`13af120`](https://github.com/rickstaa/stable-gym/commit/13af12056921614f8e7274c1aa17ed21ecaa763a)
- :bookmark: Updates code version to v0.4.2 [`5b8d0af`](https://github.com/rickstaa/stable-gym/commit/5b8d0afe86bc07b0ba8fcf15773c89bfd87a659c)
- :bookmark: Updates code version to v0.4.1 [`998aa9b`](https://github.com/rickstaa/stable-gym/commit/998aa9be295a61ba4bae04e2833fd99518cca2c6)
- :bookmark: Updates code version to v0.4.0 [`46ae85e`](https://github.com/rickstaa/stable-gym/commit/46ae85ec8b521def5862c41f8f83f39f38873d22)
- :bookmark: Updates code version to v0.3.7 [`6098739`](https://github.com/rickstaa/stable-gym/commit/60987396fd2758000644746e0b4a5f224ce9d37d)
- :bookmark: Updates code version to v0.3.6 [`bd36baa`](https://github.com/rickstaa/stable-gym/commit/bd36baa7aaf35f6910f40f6ba6d8f609cc002acb)
- :bookmark: Updates code version to v0.3.5 [`afb5fd5`](https://github.com/rickstaa/stable-gym/commit/afb5fd5e23c78de7476f7b7fe684823747bef543)
- :bookmark: Updates code version to v0.3.4 [`d317019`](https://github.com/rickstaa/stable-gym/commit/d317019ec264ee63318e2097035569b5518abab7)
- :bookmark: Updates code version to v0.3.2 [`488b0e2`](https://github.com/rickstaa/stable-gym/commit/488b0e2d5240500ebd557af01bdb8f5511392879)
- :bookmark: Updates code version to v0.3.1 [`18574a6`](https://github.com/rickstaa/stable-gym/commit/18574a648d8f1506066700befceaaa4f3820cb09)
- :bookmark: Updates code version to v0.3.0 [`bfe3a6c`](https://github.com/rickstaa/stable-gym/commit/bfe3a6c55dc0d7bd460f4dee15d07108a349079c)
- :bookmark: Updates code version to v0.2.7 [`61f2edf`](https://github.com/rickstaa/stable-gym/commit/61f2edf095dc746d6b5602218fa6095709b0c514)
- :bookmark: Updates code version to v0.2.6 [`88829be`](https://github.com/rickstaa/stable-gym/commit/88829becd7debd70ab41b17a32bf4cfe2a75a8b6)
- :bookmark: Updates code version to v0.2.5 [`d3f5b4e`](https://github.com/rickstaa/stable-gym/commit/d3f5b4ea660c890db531474e8a0dc10a9f49f5b4)
- :memo: Updates changelog [`23d2fc6`](https://github.com/rickstaa/stable-gym/commit/23d2fc6cebc05b0844ad7377b53326d5f453e5a6)
- :memo: Updates changelog [`38b9105`](https://github.com/rickstaa/stable-gym/commit/38b91055adeefc660eb9401263968fee0442d18a)
- :memo: Updates changelog [`ce26868`](https://github.com/rickstaa/stable-gym/commit/ce26868356848e63e9e9e937e0b88802208fcb29)
- :memo: Updates changelog [`eb62904`](https://github.com/rickstaa/stable-gym/commit/eb629047ba6bee04b367c9f0af3bbd96de9ea072)
- :memo: Updates changelog [`5de6eeb`](https://github.com/rickstaa/stable-gym/commit/5de6eeba8481eae051836e287f8461a6f24fcfa4)
- :memo: Updates CHANGELOG.md [`7e3ba63`](https://github.com/rickstaa/stable-gym/commit/7e3ba6342480976138570438849f626507bf4738)
- :bookmark: Bump version: 0.1.6 → 0.1.7 [`b2f821e`](https://github.com/rickstaa/stable-gym/commit/b2f821efb2a92dc62cec69416654df9831fceea6)
- :art: Improves code formatting [`3a8b688`](https://github.com/rickstaa/stable-gym/commit/3a8b688bae33c31f8766d1d430be55f7674540c5)
- :fire: Removes envs README.md [`8c23119`](https://github.com/rickstaa/stable-gym/commit/8c2311985b8ad14ae26ca935b8bce517b2826e20)
- :green_heart: Updates gh-actions and submodules [`ca364df`](https://github.com/rickstaa/stable-gym/commit/ca364df1a6a3cbd96283bf5061c9b148b40a9142)
- :bookmark: Bump version: 0.2.1 → 0.2.2 [`7c043dc`](https://github.com/rickstaa/stable-gym/commit/7c043dc5085bcabea3cedd1b3a2031e06344e48e)
- :bookmark: Bump version: 0.2.0 → 0.2.1 [`e855b4e`](https://github.com/rickstaa/stable-gym/commit/e855b4e6cabaa1f13467e9682f066ab543406046)
- :bookmark: Fixes version tags [`47fa195`](https://github.com/rickstaa/stable-gym/commit/47fa195d2f3e0beb4ebb0a83344fa34e60b2cc43)
- :bookmark: Bump version: 0.1.8 → 0.1.9 [`87fe1e1`](https://github.com/rickstaa/stable-gym/commit/87fe1e16c73daae8ea264803cfce587ef240edb3)
- :bookmark: Bump version: 0.1.7 → 0.1.8 [`759504d`](https://github.com/rickstaa/stable-gym/commit/759504d69cdcd5b9a6fbd71edcc72bdd32bc0871)
- :green_heart: Removes latest tag [`7b3fa4f`](https://github.com/rickstaa/stable-gym/commit/7b3fa4f758f53695e4df5e8782003deae2549fd4)
- :green_heart: Adds latest tag to release gh-action [`374cb1e`](https://github.com/rickstaa/stable-gym/commit/374cb1ed6bc1fea3a6aa279148c15412842a06d4)
- :white_check_mark: Updates tests [`1aea7fb`](https://github.com/rickstaa/stable-gym/commit/1aea7fbd24196366be4fdc8a215415add9287ece)
- :bug: Fixes missing dependencies for the github actions [`225b8b2`](https://github.com/rickstaa/stable-gym/commit/225b8b2a93f1e57e8b92b5cdd70131af6c033f15)
- Add renovate.json [`909aa24`](https://github.com/rickstaa/stable-gym/commit/909aa24f5fcf9d024a62c5097ba5298c72ea69b8)
- :wrench: Updates pyproject.toml file [`c1ba089`](https://github.com/rickstaa/stable-gym/commit/c1ba0899435c1c6208e86fb5a9f5017729438b84)
- :alien: Updates environment submodules [`c07800e`](https://github.com/rickstaa/stable-gym/commit/c07800eea43a5f8fd73883ec7b3cdb6d953ad610)
- :alien: Updates versioning of the submodules [`a0f49dc`](https://github.com/rickstaa/stable-gym/commit/a0f49dc4d75e326d00a564c26e7121349fe79e5a)
- :alien: Updates submodules [`1481a35`](https://github.com/rickstaa/stable-gym/commit/1481a3514903a96502635ac47c18af5487f2f329)
- :alien: Updates submodules [`9518627`](https://github.com/rickstaa/stable-gym/commit/9518627a3976603b5299f6630b334bfe929df331)
- :alien: Updates ex3_ekf and oscillator submodules [`e19cd9d`](https://github.com/rickstaa/stable-gym/commit/e19cd9db11ae347b05927a8fd207bbd3b93e024d)
- :arrow_up: Updates submodules [`f75d735`](https://github.com/rickstaa/stable-gym/commit/f75d7350597958d552f159d959cf13038a8fa816)
- :art: Fixes some syntax errors in the submodules [`4e1bc80`](https://github.com/rickstaa/stable-gym/commit/4e1bc80ceeefea99ba6395ca2d28e54148308af7)
- :alien: Updates submodules [`f3f91fe`](https://github.com/rickstaa/stable-gym/commit/f3f91fef400fb5ed214695f5085b581c980bf15f)
- :bug: Fixes corrupted submodules [`aee5fb4`](https://github.com/rickstaa/stable-gym/commit/aee5fb4773ccd6a9b57c0b9a88102066738fc19a)
- :bulb: Updates code comments [`3fb1aba`](https://github.com/rickstaa/stable-gym/commit/3fb1aba14c7efa899e16d0090f1c6776c780de12)
- :bookmark: Fixes versioning [`a10010c`](https://github.com/rickstaa/stable-gym/commit/a10010c7145b42b8113f03482defe0d3e5006b80)
- :sparkles: Updates oscillator environment [`56d9d65`](https://github.com/rickstaa/stable-gym/commit/56d9d653cc4a035cc35b7c1938e785aa6434d9b2)
- :memo: Updates README.md [`f143c27`](https://github.com/rickstaa/stable-gym/commit/f143c27496c663d61c01615c1fc340ccf62257f5)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`04f2b85`](https://github.com/rickstaa/stable-gym/commit/04f2b85101cd71bc4f47087513781233f75086f7)
- :twisted_rightwards_arrows: Merge branch 'adds_release_gh_action' into main [`c54ffab`](https://github.com/rickstaa/stable-gym/commit/c54ffab1bd54e8f198fb625fb9cb4b013daba567)
- :green_heart: Fixes gh-action names [`a0da4fe`](https://github.com/rickstaa/stable-gym/commit/a0da4fe249d73bc526bbd9531dba960127cfc571)
- :green_heart: Updates gh-action [`e21b53f`](https://github.com/rickstaa/stable-gym/commit/e21b53f743c0bd8a6095b2cb9ec06090c31fc8b4)
- :green_heart: Updates stable-gym gh-action [`78378ee`](https://github.com/rickstaa/stable-gym/commit/78378ee05d11d9a96317808c238380662469fc66)
- :arrow_up: Merge pull request #3 from rickstaa/renovate/matplotlib-3.x [`5385ab3`](https://github.com/rickstaa/stable-gym/commit/5385ab3a00db941cb24eedc24d57b5057c2a070d)
- :arrow_up: Updates submodules environments [`ed608a3`](https://github.com/rickstaa/stable-gym/commit/ed608a34be90ae95e866a4231ba313b0ad43b17a)
- :bookmark: Bump version: 0.1.4 → 0.1.5 [`b6c8a51`](https://github.com/rickstaa/stable-gym/commit/b6c8a51bc41c76cf81f00f676e1d5d2d4b1e4472)
- :bookmark: Bump version: 0.1.3 → 0.1.4 [`6e4917e`](https://github.com/rickstaa/stable-gym/commit/6e4917ef9b37e9eded87126fd7b171c9ce544edc)
- :bookmark: Bump version: 0.1.2 → 0.1.3 [`5a4261d`](https://github.com/rickstaa/stable-gym/commit/5a4261dc2e5f2e178e73fab40109ac6c59d42bd7)
- :art: Add __version__ variable [`3e2dbed`](https://github.com/rickstaa/stable-gym/commit/3e2dbed2b737e685cc56fd0851d66af990e5f7f7)
- Update dependency matplotlib to v3.3.3 [`310135d`](https://github.com/rickstaa/stable-gym/commit/310135dabd47292212746b3e08df0b50da29c024)
- :arrow_up: Merge pull request #2 from rickstaa/renovate/gym-0.x [`aa4ba0f`](https://github.com/rickstaa/stable-gym/commit/aa4ba0fe07508c2bdf4b7d45c6ca416b2b60b9f2)
- Update dependency gym to v0.18.0 [`105c4ae`](https://github.com/rickstaa/stable-gym/commit/105c4ae1548b7500121f1095739ac14f93c71ff8)
- :construction_worker: Merge pull request #1 from rickstaa/renovate/configure [`2a7e126`](https://github.com/rickstaa/stable-gym/commit/2a7e126da2e626f9340aad2b5afcc07daf3a35c7)
- :up_arrow: Updates submodules [`1379584`](https://github.com/rickstaa/stable-gym/commit/13795848cabb263f521074bff6262cbbc8ac753f)
- :wrench: Updates pip package url [`3564e12`](https://github.com/rickstaa/stable-gym/commit/3564e120cb44a87ed8f329da0a63ad92c63a4de2)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`029e2bf`](https://github.com/rickstaa/stable-gym/commit/029e2bfc4d48f8817ef2a3714e5a827ce7e79c65)
- :sparkles: Updates oscillator environment [`696fa9d`](https://github.com/rickstaa/stable-gym/commit/696fa9d5c8c2ad85c6433cb38f1e7fdfe310c522)
- :package: Changes package to namespace package [`2f13f1e`](https://github.com/rickstaa/stable-gym/commit/2f13f1e01d68f4a3e34a549ddbc88291b0fe247a)
- :sparkles: Adds Ex3_EKF environment git submodule [`f6ac9ff`](https://github.com/rickstaa/stable-gym/commit/f6ac9ff968e4ec527eb1d7e4ca357b4e43a33408)
- :sparkles: Updates oscillator submodule [`ce19923`](https://github.com/rickstaa/stable-gym/commit/ce1992340c540a694802595f80ccca1ccbc774bb)
- :memo: Updates README.md [`38a4dc4`](https://github.com/rickstaa/stable-gym/commit/38a4dc4edaa87effc950233f05bd4c7d365f598f)
- :arrow_up: Updates setup.py file and environment registration method [`55b5128`](https://github.com/rickstaa/stable-gym/commit/55b51288572d7648e398131c68f7727077f6833f)
- :wrench: Updates setuptools cfg file [`7386a49`](https://github.com/rickstaa/stable-gym/commit/7386a49a1fc3919df9c2071cb30f224e25a12927)
- :memo: Updates CHANGELOG [`252532c`](https://github.com/rickstaa/stable-gym/commit/252532c61070da91d5fff0e9e0ff646dfd507174)
- page_facing_up: Updates CHANGELOG [`9019d70`](https://github.com/rickstaa/stable-gym/commit/9019d703ba969a84347bfe908662f775f26906f2)
- :alien: Updates Oscillator environment version [`9450234`](https://github.com/rickstaa/stable-gym/commit/9450234c4971b1592c84d7ba75681d49306aacbc)
- :pencil: Updates README [`4281343`](https://github.com/rickstaa/stable-gym/commit/42813436f6045a0fd36acb06e50130e126e7bd39)
- :alien: Updates environment submodules [`e14184c`](https://github.com/rickstaa/stable-gym/commit/e14184c5fa3f1cfdbadb87c3d445dc02b60b76aa)
- :white_check_mark: Updates tests [`99f7f03`](https://github.com/rickstaa/stable-gym/commit/99f7f03c7f98b4619b570108954ecc7535e584bc)
- :heavy_minus_sign: Depricate setuptools_scm and use bump2version instead [`1ee46ce`](https://github.com/rickstaa/stable-gym/commit/1ee46ce732284d2aaa04e9e63badecb61597b981)
- :bug: Fixes bug in the pyproject.toml [`ee268ae`](https://github.com/rickstaa/stable-gym/commit/ee268aeb7e97770215bc842a099e80d42bef5191)
- :bookmark: Bump version: 0.0.2 → 0.0.3 [`9288f7f`](https://github.com/rickstaa/stable-gym/commit/9288f7ff96e50dc1daafbc8b2d1e36695a01ac3e)
- :pencil: Updates readme.md [`340ae90`](https://github.com/rickstaa/stable-gym/commit/340ae902cfcf6a0f6484c49617c0519e645c72fc)
- :pencil: Updates readme [`66596c1`](https://github.com/rickstaa/stable-gym/commit/66596c17a2b882b8b6c9f86e2fdb553291d6eb62)
- :bookmark: Bump version: 0.1.1 → 0.1.2 [`631f76d`](https://github.com/rickstaa/stable-gym/commit/631f76dfc7065f3c2d7cf6a79df724c63ecc22c9)
- :alien: Updates environment submodules [`ec58e44`](https://github.com/rickstaa/stable-gym/commit/ec58e440411700ce0cefefc6fcc790b1ad310216)
- :bookmark: Bump version: 0.1.0 → 0.1.1 [`5f7f42c`](https://github.com/rickstaa/stable-gym/commit/5f7f42cbe7826e7ed93dfedaa0ddb0e90f7738db)
- :twisted_rightwards_arrows: Merge branch 'main' of github.com:rickstaa/stable-gym into main [`77f2024`](https://github.com/rickstaa/stable-gym/commit/77f20249f1a4a15be6ea2efd95dbb6ded71dcc72)
- :bookmark: Bump version: 0.0.3 → 0.1.0 [`747ed41`](https://github.com/rickstaa/stable-gym/commit/747ed41b603e76798083ca27485462092d36e418)
- :memo: Updates README.md [`e6b96ab`](https://github.com/rickstaa/stable-gym/commit/e6b96ab8c4a638320b273860626e6d4641cd50b4)
- :memo: Updates README.md [`1656e45`](https://github.com/rickstaa/stable-gym/commit/1656e45347fdca00536786dbbcda50fdb104fe5f)
- :alien: Updates environment submodules [`ca1ead1`](https://github.com/rickstaa/stable-gym/commit/ca1ead19cd8b05d92f6981090151e11fa7e1b2c1)
- :wrench: Updates bumpversion configuration file [`2423841`](https://github.com/rickstaa/stable-gym/commit/24238418f3f25917900b1cab4ea24c89b3a9f5cd)
- :sparkles: Updates oscillator environment [`635184a`](https://github.com/rickstaa/stable-gym/commit/635184add4444696b18bf8f4ceb3bdec90fa61b0)
- :bug: Updates oscillator submodule [`87296d0`](https://github.com/rickstaa/stable-gym/commit/87296d0cd0b0cdfa656bf01631e9c3c885b0e88d)
- :bug: Updates oscillator module [`96f3b6d`](https://github.com/rickstaa/stable-gym/commit/96f3b6d44397a390618aab630ff9ee7060ff2264)
- :sparkles: Updates oscillator submodule [`953982e`](https://github.com/rickstaa/stable-gym/commit/953982ed5b519ef8dcb7679722603029fdcb393a)
- :wrench: Adds setup.cfg to bumpversion files [`f2c55ca`](https://github.com/rickstaa/stable-gym/commit/f2c55ca0ea17711eaf36649afd513344e1571e23)
- :fire: Removes unused files [`1f6ae24`](https://github.com/rickstaa/stable-gym/commit/1f6ae243149e6c4834733af3ac9e333c22968bd5)

## v0.0.0 - 2020-08-14

### Commits

- :tada: Initial commit [`a2ffda3`](https://github.com/rickstaa/stable-gym/commit/a2ffda399e812814a79fe6b73b0629dff99ad77c)
