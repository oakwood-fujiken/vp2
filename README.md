# VP<sup>2</sup>

VP<sup>2</sup> is a benchmark for video prediction models for robotic manipulation via model-predictive control.
This code accompanies the paper [A Control-Centric Benchmark for Video Prediction](https://arxiv.org/abs/2304.13723)
(ICLR 2023). The project page containing more details can be found [here](https://s-tian.github.io/projects/vp2/).

## Installation

VP<sup>2</sup> is a regular pip-installable Python package targeting **Python 3.10+**.
A single `pip install` resolves every runtime dependency — there are **no extra
manual `pip install` steps** for the simulator stack, and the previously-pinned
`s-tian` forks of `robosuite`, `robomimic` and `iGibson` are no longer used.

```
pip install git+https://github.com/s-tian/vp2.git
```

or, after cloning,

```
pip install .
```

This installs the **upstream / official** releases of the simulator stack:

| Package      | Source                                                              |
| ------------ | ------------------------------------------------------------------- |
| `robosuite`  | PyPI (`>=1.4.1`)                                                    |
| `robomimic`  | PyPI (`>=0.3.0`)                                                    |
| `gymnasium`  | PyPI (`>=0.28`) — replaces the abandoned `gym` package              |
| `dm_control` | PyPI (`>=1.0.10`)                                                   |
| `mujoco`     | PyPI (`>=2.3.3`)                                                    |
| `robodesk`   | `git+https://github.com/google-research/robodesk.git` (no PyPI release) |

The fork-only extensions used by the original VP<sup>2</sup> code (custom
domain randomisation, depth/segmentation/normal plumbing, the
`get_object_positions` helper) are reimplemented inside the package itself in
[`vp2/envs/_robosuite_compat.py`](vp2/envs/_robosuite_compat.py); they sit on
top of the public APIs of the upstream packages, so no patched library is
required.

> **Removed feature:** the legacy `renderer="igibson"` path, which depended on
> the `s-tian/robomimic@depth_images` + `s-tian/iGibson` patches, is no longer
> available. VP<sup>2</sup> now defaults to robosuite's built-in MuJoCo
> renderer (`env.renderer=mujoco`); selecting `igibson` will emit a warning
> and transparently fall back to MuJoCo. The original Python-3.7 conda
> environment (`environment.yml`) is still shipped for users who need to
> reproduce the iGibson-based pipeline exactly.

### Optional video-prediction model integrations

The four pretrained video-prediction backbones are kept as separate Python
packages and exposed as install extras. Install whichever you need:

```
pip install "vp2[fitvid] @ git+https://github.com/s-tian/vp2.git"
pip install "vp2[svg_prime] @ git+https://github.com/s-tian/vp2.git"
pip install "vp2[mcvd] @ git+https://github.com/s-tian/vp2.git"
pip install "vp2[struct_vrnn] @ git+https://github.com/s-tian/vp2.git"
```

Multiple extras can be combined: `vp2[fitvid,svg_prime]`. A `dev` extra is also
available with `ipdb` and `pytest`.

### Datasets and pretrained weights

You will also need to download the data containing task instance specifications
(initial states and goals) as well as the classifier weights for the robodesk
environment. Pretrained models for SVG', FitVid, MCVD, and StructVRNN are also
included.

These can be found at: https://purl.stanford.edu/qf310mj0842 (19.6GB), and on
HuggingFace at
https://huggingface.co/datasets/s-tian/VP2/tree/main/vp2_benchmark_data.

Because VP<sup>2</sup> is now an installed package, you only need the *data*
folders locally — they no longer need to live next to the source tree. The
recommended layout for a working directory is:

```
my-vp2-runs/
├── cost_classifiers/
├── robodesk_benchmark_tasks/
├── robosuite_benchmark_tasks/
└── pretrained_models/
```

The Hydra configs reference these via `goals_dataset` (see
`vp2/scripts/configs/env/robosuite.yaml` and
`vp2/scripts/configs/env/robodesk.yaml`) and `model.checkpoint_dir` (per-model
configs under `vp2/scripts/configs/model/`). Paths are resolved relative to the
directory the script is launched from. Override them on the command line if
you store the data elsewhere, e.g.:

```
vp2-run-control env.goals_dataset=/data/vp2/robosuite_benchmark_tasks/combined/rendered_256.hdf5
```

Lastly, video-prediction training datasets for both the robosuite and robodesk
environments can be found at https://purl.stanford.edu/cw843qn4148 (182GB
total) and on HuggingFace at
https://huggingface.co/datasets/s-tian/VP2/tree/main/vp2_training_datasets.
Note that while there are separate data files for different robodesk tasks, we
always train models on all of the robodesk data (or all the robosuite data) at
once.

## Running control experiments

Configs in this repo are handled using [Hydra](https://hydra.cc/).
Once installed, the entry point is the `vp2-run-control` console script
(equivalent to `python -m vp2.scripts.run_control`):

```
vp2-run-control                                # use the defaults
vp2-run-control model=svg_prime env=robodesk   # Hydra overrides
```

This will run a control benchmark using the configuration specified in
`vp2/scripts/configs/config.yaml`. The high-level configuration choices are:

- `model`: the video prediction model to use. Default `fitvid`. To switch, set
  e.g. `model=svg_prime` (config files live in
  `vp2/scripts/configs/model/`).
- `agent`: default `planning_agent` (MPC). `random_agent` is a random baseline.
- `env`: default `robosuite`. `robodesk` covers the remaining tasks.

Lower level configuration choices include:

- `agent/optimizer`: optimiser used by the MPC agent. Default `mppi`. We
  implement `cem`, `mppi`, `cem-gd`, and `lbfgs`. The latter two require the
  model and cost function to be differentiable.

## Example control experiment commands

We provide the experiment commands used in the case study in the paper in the
`experiments` folder. Update the entry point in those commands from
`python scripts/run_control.py` to `vp2-run-control` to use the installed
console script.

## Set up video prediction models

Each backbone is its own Python package; the simplest install path is the
extras above. Manual installs are equally fine if you want to develop against a
local clone:

### SVG'

```
pip install "vp2[svg_prime]"
# or, for a development checkout:
git clone https://github.com/s-tian/svg-prime.git
pip install -e ./svg-prime
```

Example commands for control experiments can be found in
`experiments/case_study.txt`.

### FitVid

```
pip install "vp2[fitvid]"
# or
git clone https://github.com/s-tian/fitvid.git
pip install -e ./fitvid
```

Example commands for control experiments can be found in
`experiments/case_study.txt`. Unfortunately the exact model weights for the
FitVid model used in the paper case study are not available, but the weights
provided in the pretrained download are trained in the same way as the models
in the paper.

### MCVD

```
pip install "vp2[mcvd]"
# or
git clone https://github.com/s-tian/mcvd-pytorch.git
pip install -e ./mcvd-pytorch
```

Example command to run control with MCVD on the robodesk environment
(`push_red` task):

```
vp2-run-control hydra.job.name=test_mcvd planning_modalities=[rgb] seed=0 env=robodesk \
  agent.optimizer.init_std=[0.5,0.5,0.5,0.1,0.1] env.task=push_red \
  model=mcvd model_name=mcvd \
  agent.optimizer.objective.objectives.rgb.weight=0.5 \
  agent.optimizer.objective.objectives.classifier.weight=10 \
  agent/optimizer/objective=combined_classifier_mse agent.optimizer.log_every=5 \
  model.checkpoint_dir=pretrained_models/mcvd/rdall_base/checkpoint_330000.pt
```

### Struct-VRNN

```
pip install "vp2[struct_vrnn]"
# or
git clone https://github.com/s-tian/struct-vrnn-pytorch.git
pip install -e ./struct-vrnn-pytorch
```

Example command to run control with Struct-VRNN on the robodesk environment
(`push_red` task):

```
vp2-run-control hydra.job.name=test_structvrnn model_name=structvrnn_robodesk \
  model.checkpoint_dir=pretrained_models/struct-vrnn/rdall_base/ model.epoch=210000 \
  model=keypoint_vrnn planning_modalities=[rgb] seed=0 env=robodesk \
  agent.optimizer.init_std=[0.5,0.5,0.5,0.1,0.1] env.task=push_red \
  agent.optimizer.objective.objectives.rgb.weight=0.5 \
  agent.optimizer.objective.objectives.classifier.weight=10 \
  agent/optimizer/objective=combined_classifier_mse agent.optimizer.log_every=5
```

Note that some large-valued pixel "specks" appear in the Struct-VRNN
predictions. This is a so far unexplained artifact of the model, that may be
due to my reimplementation.

### MaskViT

Please stay tuned for the MaskViT code release
[here](https://github.com/agrimgupta92/maskvit).

## Adding new models

To add a new model, you can add a new config file to the
`vp2/scripts/configs/model` folder. The config file should use `_target_` to
specify the model class to use. The remainder of config items will be passed to
the model class as kwargs.

The model class should implement a function `__call__` that takes as input a
dictionary with two keys:

- `video`: context frames from the environment, a torch tensor of shape
  `(B, T, C, H, W)` in range `[0, 1]`
- `actions`: the history of actions taken by the agent, a torch tensor of shape
  `(B, T, A)` in range `[-1, 1]`

where `B` is the batch size, `T` is the number of frames in the video, `C` is
the number of channels, `H` and `W` are the height and width, and `A` is the
action dimension.

It should then return a dictionary with one key:

- `rgb`: a torch tensor of shape `(B, T+n_pred, C, H, W)` in range `[0, 1]`
  containing the predicted frames.

## Adding new environments

Although we provide code and configs for the robosuite and robodesk
environments, it is relatively straightforward to add new environments. To add
a new environment, you should do the following:

- Create a new environment class implementing the `BaseEnv` interface in
  [`vp2/envs/base.py`](vp2/envs/base.py).
- Add a new config file to the `vp2/scripts/configs/env` folder. This should
  specify the environment class to use and any parameters to pass to the
  environment class.
- Use the `env` config to specify the new environment from the `env` config
  group in the control experiment config or via a command-line override.

## Rendering low-dimensional datasets

The full (50k trajectory) training dataset for the robosuite environment is
provided as `robosuite_demo_1` through `robosuite_demo_5`. Each dataset
contains 10000 trajectories (so there are 50000 in total). Note that we
perform most experiments on just 5000 trajectories, which is provided with
full RGB image data. The download for the full 50k datasets only contains the
raw environment observations.

To perform video prediction training, they must be rendered into images. The
original release rendered them with the
[`s-tian/robomimic@depth_images`](https://github.com/s-tian/robomimic/tree/depth_images)
fork, which added an `igibson` renderer option. With the upstream-only setup,
use the public `robomimic` script with `--renderer mujoco`:

```
python -m robomimic.scripts.dataset_states_to_obs \
  --dataset /PATH/HERE/demo.hdf5 \
  --output_name rendered_256.hdf5 \
  --done_mode 2 \
  --camera_names agentview_shift_2 \
  --camera_height 256 --camera_width 256 \
  --renderer mujoco
```

(This renders the data at `256x256` resolution as we do in the paper; specify
any resolution you like.) Reproducing the iGibson-rendered images bit-for-bit
still requires the original fork.

## FAQ / Troubleshooting

#### MuJoCo rendering fails with the message `Offscreen framebuffer is not complete, error 0x8cdd`:
This seems to be related to EGL driver issues with the `dm_control` package.
See [this thread](https://github.com/deepmind/dm_control/issues/370) for more
details. For a workaround, try setting the `dm_control` rendering environment
variable via `export MUJOCO_GL=osmesa`. Note that this unfortunately does not
support GPU rendering, but this is usually not the bottleneck for visual
foresight experiments.

#### `hydra.errors.MissingConfigException: Primary config module 'vp2.scripts.configs' not found`
This used to occur when `vp2` was installed as a wheel because the configs
directory had no `__init__.py` files. Fixed in this branch — make sure you are
running an up-to-date install (`pip install --upgrade .`).

#### `hydra.errors.InstantiationException: Error locating target 'vp2.models...'`
This error occurs when the model class specified in the config file cannot be
found. Make sure the corresponding model package is installed (see
[Set up video prediction models](#set-up-video-prediction-models)).

#### `renderer='igibson' is only supported via the s-tian/robomimic depth_images fork`
This is a warning, not an error. The upstream-only build no longer ships an
iGibson renderer; VP<sup>2</sup> automatically falls back to robosuite's
built-in MuJoCo renderer. Set `env.renderer=mujoco` explicitly to silence the
warning.

## Citation

If you find this code useful, please cite the following paper:

```
@inproceedings{tian2023vp2,
  title={A Control-Centric Benchmark for Video Prediction},
  author={Tian, Stephen and Finn, Chelsea and Wu, Jiajun},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
