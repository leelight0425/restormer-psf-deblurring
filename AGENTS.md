# Agent Instructions

## Layout

- Run commands from the repository root; relative paths in YAML are resolved from the current working directory.
- This is a BasicSR-style PyTorch project. `basicsr/train.py` and `basicsr/test.py` are the main entrypoints.
- Dataset and model modules are auto-discovered from filenames ending in `_dataset.py` and `_model.py`; new classes do not need a manual registry edit.
- PSF experiments and their YAML files live under `PSF_Deblurring/`; the corresponding dataset and model code lives under `basicsr/`.

## Setup And Commands

- Install the dependencies listed in `INSTALL.md` first. `setup.py` returns an empty runtime requirement list, so package installation does not install the required dependencies.
- Install the package with `python setup.py develop --no_cuda_ext` when CUDA extensions are not needed.
- Run single-process training with `python basicsr/train.py -opt <train-config.yml>`.
- `bash train.sh <train-config.yml>` always launches 8 distributed processes on port 4321 and requires a POSIX shell plus 8 GPUs.
- Use `python basicsr/test.py -opt <test-config.yml>` for simulated PSF evaluation with a YAML dataset and GT images; do not use it for real captured photos.
- Process real captured photos with `python inference_psf.py --opt <config.yml> --input <photo> --output <result>`; the script tiles the image and derives XY/PSF data from each tile position.
- Evaluate checkpoints with `python PSF_Deblurring/batch_eval.py --opt <test-config.yml> --models_dir <models-dir> --every <N>`; checkpoints must use the `net_g_<iter>.pth` naming pattern.
- No repository test suite or CI runner is configured. For focused Python changes, run `python -m py_compile <changed.py>` and `git diff --check`.
- Training automatically resumes the highest-numbered `.state` file under `experiments/<name>/training_states`; use a new experiment name or deliberately manage that directory for a fresh run.
- The options parser injects dataset `phase` from the YAML section name and expands only `dataroot_gt` and `dataroot_lq`; keep `npz_dir` paths valid from the repository root.

## PSF Invariants

- `Dataset_PSFDeblurring` and `Dataset_PSFDeblurringGray` expect `.npz` files containing `psf`, `cx`, and `cy`, support disk or LMDB GT input, and return `psf_kernel` in HWC form for the Restormer PSF encoder.
- When `kernel_channels > 0`, Restormer's PSF encoder is a submodule of `net_g`; its weights are saved in the same `net_g_<iter>.pth` checkpoint, as in NAFNet.
- Training flips and rotations must be applied to the full GT image before crop selection, PSF selection, convolution, and XY generation; changing this order makes the blur and sensor coordinates inconsistent.
- Validation edge sampling is controlled by `psf_edge_threshold` (the PSF YAMLs use `500`; `0` disables it); a random point in an edge strip is sampled first and `_find_nearest_psf` selects its kernel instead of using a fixed image-center crop.
- Keep `use_xy` consistent with `network_g.inp_channels` (`3` for RGB or `5` for RGB plus XY), and keep `kernel_channels > 0` paired with the `psf_kernel` data path.
- PSF YAML files contain machine-specific dataset, PSF, and checkpoint paths; update those paths before running an experiment.
