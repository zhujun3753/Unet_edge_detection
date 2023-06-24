# DexiNed with Unet Version

- Original [README file](./README_old.md).
- Dataset for training: [BIPEDv2](https://xavysp.github.io/MBIPED/)
- Run:
  - Download `BIPEDv2` from [here](https://xavysp.github.io/MBIPED/)
  - BIPED Data Augmentation, **make sure** that `dataset_dir` is the directory of `BIPEDv2`.
    ```shell
    python MBIPED/main.py
    ```
  - Training
    ```shell
    python main.py --is_testing=0 --use_unet=1
    ```
  - Test
    ```shell
    python main.py --is_testing=1 --use_unet=1
    ```