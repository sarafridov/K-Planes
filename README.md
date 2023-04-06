# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

Where we develop an extensible (to arbitrary-dimensional scenes) and explicit radiance field model which can be used for static, dynamic, and variable appearance datasets.

Code release for:

> __K-Planes: Explicit Radiance Fields in Space, Time, and Appearance__
>
> [Sara Fridovich-Keil*](https://people.eecs.berkeley.edu/~sfk/), [Giacomo Meanti*](https://www.iit.it/web/iit-mit-usa/people-details/-/people/giacomo-meanti), [Frederik Rahbæk Warburg](https://frederikwarburg.github.io/), [Benjamin Recht](https://people.eecs.berkeley.edu/~brecht/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/)

:rocket: [Project page](https://sarafridov.github.io/K-Planes)

:newspaper: [Paper](https://arxiv.org/abs/2301.10241)

:file_folder: [Raw output videos and pretrained models](https://drive.google.com/drive/folders/1zs_folzaCdv88y065wc6365uSRfsqITH)



## Setup 

We recommend setup with a conda environment using PyTorch for GPU (a high-memory GPU is not required). Training and evaluation data can be downloaded from the respective websites (NeRF, LLFF, DyNeRF, D-NeRF, Phototourism). 

## Training

Our config files are provided in the `configs` directory, organized by dataset and explicit vs. hybrid model version. These config files may be updated with the location of the downloaded data and your desired scene name and experiment name. To train a model, run
```
PYTHONPATH='.' python plenoxels/main.py --config-path path/to/config.py
```

Note that for DyNeRF scenes it is recommended to first run for a single iteration at 4x downsampling to pre-compute and store the ray importance weights, and then run as usual at 2x downsampling. This is not required for other datasets.

## Visualization/Evaluation

The `main.py` script also supports rendering a novel camera trajectory, evaluating quality metrics, and rendering a space-time decomposition video from a saved model. These options are accessed via flags `--render-only`, `--validate-only`, and `--spacetime-only`, and a saved model can be specified via `--log-dir`.


## License and Citation

```
@inproceedings{kplanes_2023,
      title={K-Planes: Explicit Radiance Fields in Space, Time, and Appearance},
      author={{Sara Fridovich-Keil and Giacomo Meanti} and Frederik Rahbæk Warburg and Benjamin Recht and Angjoo Kanazawa},
      year={2023},
      booktitle={CVPR}
}
```
Note: Joint first-authorship is not fully supported in BibTex; you may need to modify the above depending on your format.

This work is made available under the BSD 3-clause license. Click [here](LICENSE) to view a copy of the license.
