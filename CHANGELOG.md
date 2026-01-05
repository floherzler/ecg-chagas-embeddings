# Changelog

## [0.3.0](https://github.com/floherzler/ecg-chagas-embeddings/compare/v0.2.0...v0.3.0) (2026-01-05)


### Features

* add experiment scripts for tracks 1 and 2 ([f00f901](https://github.com/floherzler/ecg-chagas-embeddings/commit/f00f901912751038031b2c79d0c2f0821563c79e))
* add metrics and vis script from TTC paper ([51ddb21](https://github.com/floherzler/ecg-chagas-embeddings/commit/51ddb2192b590cb59204ef0a6297baa312c03b67))
* add QC metrics (after bandpass) from neurokit2 to metadata ([7a06a91](https://github.com/floherzler/ecg-chagas-embeddings/commit/7a06a91c2eea5673d89ff2d093b8ebea7f1ab9f4))
* add umap logging each epoch to wandb ([a6c4bce](https://github.com/floherzler/ecg-chagas-embeddings/commit/a6c4bcea77927fc7f4452796aa04cceef33bccd0))
* added experiment tracks and model architecture changes based on this, created configs and shell scripts ([5038b0a](https://github.com/floherzler/ecg-chagas-embeddings/commit/5038b0add83ad84349e3fc6ffe9b63a578d960bf))
* augmentations depend on train/val to keep UMAP consistent; improve UMAP plotting and add LinearClassifier from TTC paper ([8d13ba1](https://github.com/floherzler/ecg-chagas-embeddings/commit/8d13ba123fd313dd9d54e38890fc0bb5297921b7))
* configure all training parameters from config file ([2832170](https://github.com/floherzler/ecg-chagas-embeddings/commit/283217057630ed736b198c85fdb9775f9f888df9))
* dataset processing and neurokit explorations ([77fea6c](https://github.com/floherzler/ecg-chagas-embeddings/commit/77fea6cf9ea67e415099ade2331a5a1ab6617f89))
* improve datasets notebook for fold comparison ([9559272](https://github.com/floherzler/ecg-chagas-embeddings/commit/95592723b05cf2a543c4eb8833ca6dc99e6c0d18))
* improved data preprocessing pipeline ([e17c0cf](https://github.com/floherzler/ecg-chagas-embeddings/commit/e17c0cffd3a668b92b744dd154340646e4b83d79))
* use neurokit2 clean biosppy filter cutoffs and visualize in neurokit.ipynb ([21fef71](https://github.com/floherzler/ecg-chagas-embeddings/commit/21fef71ebabb77cc60d3335e5bc1843b97c766bd))
* vcg heart axis rotation ([002ea2f](https://github.com/floherzler/ecg-chagas-embeddings/commit/002ea2f5e62ec3ff0ee8f5f77ef9bc4e78b0e2d1))


### Bug Fixes

* arg names ([f629fa0](https://github.com/floherzler/ecg-chagas-embeddings/commit/f629fa06f481bc62c6cd4b3178f7159aaee928b1))
* correctly export paths ([c5b6fc6](https://github.com/floherzler/ecg-chagas-embeddings/commit/c5b6fc67cefb1216c658cb2cda71e0d25f9557c0))
* extend config, fix some linting mistakes and silence some typing issues for now ([1848efe](https://github.com/floherzler/ecg-chagas-embeddings/commit/1848efe16978da97d4b28599bbdea7d165e5575d))
* move so one slurm script per experiment ([8bc4803](https://github.com/floherzler/ecg-chagas-embeddings/commit/8bc48034d7bbdac393619bcd84aced6f3d79ffb2))
* pin_memory only if cuda is available ([ebc7e04](https://github.com/floherzler/ecg-chagas-embeddings/commit/ebc7e04a9bac9ad3bb7c1d8f7976486838e32b95))
* qc metrics and dataset sanity checks ([88c5f61](https://github.com/floherzler/ecg-chagas-embeddings/commit/88c5f61fc709ef7fd220da021738791b70b4f6f4))
* remove accidental duplicate config ([7a57e50](https://github.com/floherzler/ecg-chagas-embeddings/commit/7a57e5016fc7e6c14a31e2f0b65185d89d090cd9))
* silence ruff linting errors ([c7b2634](https://github.com/floherzler/ecg-chagas-embeddings/commit/c7b2634620610f9e485e03a7d879ef6262ef6356))
* simplify training parameters ([f2d5ee6](https://github.com/floherzler/ecg-chagas-embeddings/commit/f2d5ee60cc803ab427c75dcf53505f54dcac0d12))
* ty checks ([9622615](https://github.com/floherzler/ecg-chagas-embeddings/commit/9622615f750b21c9e0c293a25574d26fdf6d5020))
* use correct file path to not break notebook ([b2cf51b](https://github.com/floherzler/ecg-chagas-embeddings/commit/b2cf51bbed3162babc540403339b992581f2e77d))
* use the same augmentation structure for classifier-only training ([6c3ea22](https://github.com/floherzler/ecg-chagas-embeddings/commit/6c3ea22ec2d96be8467183c3aa95d7e2af8edd3a))

## [0.2.0](https://github.com/floherzler/ecg-chagas-embeddings/compare/v0.1.0...v0.2.0) (2025-11-05)


### Features

* add d2 thesis overview diagram and basic installation scaffold in README ([9170357](https://github.com/floherzler/ecg-chagas-embeddings/commit/91703577162edc81dce1e5437cfc386c2062beb3))
* add dataset code from physionet challenge participation ([d9402a7](https://github.com/floherzler/ecg-chagas-embeddings/commit/d9402a706ac718dc745c99eb9412f0150e7bbaea))
* add models and training code from physionet challenge ([6ab143b](https://github.com/floherzler/ecg-chagas-embeddings/commit/6ab143b6ff3035e86bcaf1ee4f42eac7034f3bef))


### Bug Fixes

* imorts and missing argparse package,  LightningCLI now working correctly! ([c4fd695](https://github.com/floherzler/ecg-chagas-embeddings/commit/c4fd6955b98943b5f018a6e95df4c287a49bcd62))
* pass weight_decay to adamw (copilot) ([3ca89c8](https://github.com/floherzler/ecg-chagas-embeddings/commit/3ca89c87d15249396d454e31c2a0cf2da7a337b3))
* some more type checks ([bdc885e](https://github.com/floherzler/ecg-chagas-embeddings/commit/bdc885eaa62f609ce45a6c6e5635eeb67c6fe08f))
* use correct ruff ignore for src imports ([11f1b91](https://github.com/floherzler/ecg-chagas-embeddings/commit/11f1b91fdf2c5a88a29efd5cf1eda0aa0bfc0d6f))
* use new lightning imort style and setup LightningCLI ([d84cc85](https://github.com/floherzler/ecg-chagas-embeddings/commit/d84cc85c77cd45b1c38df6a9b9a7df7c4361855c))

## 0.1.0 (2025-09-24)


### Features

* add ruff and ty tools for CI ([b84c5f5](https://github.com/floherzler/ecg-chagas-embeddings/commit/b84c5f5fa8ac41c202bc0460f1ccd75536962d85))
* release please config for versioning ([dc17287](https://github.com/floherzler/ecg-chagas-embeddings/commit/dc1728790d6af90afaa7261e0895cc478b1d012b))


### Bug Fixes

* point release-please to the correct file ([e25b50d](https://github.com/floherzler/ecg-chagas-embeddings/commit/e25b50d7291b2727a5100b6def402b0e922168d3))
* releaseplease config and workflow ([62e0ba2](https://github.com/floherzler/ecg-chagas-embeddings/commit/62e0ba28fca32ceb1af8e9fe186ccb3ba9a61d5b))
* remove pytest from CI ([ab53ee2](https://github.com/floherzler/ecg-chagas-embeddings/commit/ab53ee20404d7c3b2ecabd0bea3be695925a9e7c))
* replace underscores with dashes ([834dec6](https://github.com/floherzler/ecg-chagas-embeddings/commit/834dec678590b5aae0a4854cfaa2712803f777cc))
* revert workflow file ([f8f2d3d](https://github.com/floherzler/ecg-chagas-embeddings/commit/f8f2d3d070753d56ebc618e9e6598914e0cf2b3d))
* use the token to checkout the repo for release-please ([4f705bf](https://github.com/floherzler/ecg-chagas-embeddings/commit/4f705bf5049be079495b3c40ae263d99b548259c))
