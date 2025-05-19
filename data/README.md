We use the same dataset of [EO-NeRF](https://rogermm14.github.io/eonerf/) paper. We duplicate the data here for easy access.

The raw data was collected from the following sources:
* [DFC2019 challenge](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) for the JAX_004, JAX_068, JAX_214 and JAX_260 scenes.
* [IARPA]() for the IARPA_001, IARPA_002 and IARPA_003 scenes.

The images were selected, cropped and pansharpened and placed in the corresponding `data/images/{SCENE}` directory. Then the raw RPCS were bundled-adjusted.

> [!Note]
> For each scene we train the EOGS model using just the images listed in the `data/rpcs/{SCENE}/train.txt` file.