# AnySkill: Learning Open-Vocabulary Physical Skill for Interactive Agents (CVPR 2024)
<p align="left">
    <a href='https://arxiv.org/abs/2403.12835'>
      <img src='https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arXiv&logoColor=red' alt='Paper arXiv'>
    </a>
    <a href='https://anyskill.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://www.youtube.com/watch?v=QojOdY2_dTQ'>
      <img src='https://img.shields.io/badge/Video-Youtube-orange?style=plastic&logo=Youtube&logoColor=orange' alt='Video Youtube'>
    </a>
    <a href='https://drive.google.com/drive/folders/1CKJER3CnVh0o8cwlN8a2c0kQ6HTEqvqj?usp=sharing'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-green?style=plastic&logo=Google%20Drive&logoColor=green' alt='Checkpoints'>
    </a>
</p>

[//]: # (<video src="page.mp4" controls="controls" width="1080" height="720"></video>)
![](assets/teaser.png)
**AnySkill, a novel hierarchical method that learns physically plausible interactions following open-vocabulary instructions.**

[//]: # (## Introduction)
[//]: # (![]&#40;assets/model.png&#41;)


## TODOs
- [x] Release training code.
- [x] Release the model of low-level controller.
- [ ] Release the code for transparent solution tracking.


### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```

### Low-level controller training

**[NEW]** We have provided our well-trained model of low-level controller, you can download from [this link]().


First, a CALM model can be trained to imitate a dataset of motions clips using the following command:
```
python calm/run.py
--task HumanoidAMPGetup
--cfg_env calm/data/cfg/humanoid.yaml
--cfg_train ./calm/data/cfg/train/rlg/calm_humanoid.yaml
--motion_file [Your file path]/motions.yaml
--track
```
`--motion_file` can be used to specify a dataset of motion clips that the model should imitate. 
The task `HumanoidAMPGetup` will train a model to imitate a dataset of motion clips and get up after falling.
Over the course of training, the latest checkpoint `Humanoid.pth` will be regularly saved to `output/`,
along with a Tensorboard log. `--headless` is used to disable visualizations and `--track` is used for tracking using weights and biases. If you want to view the
simulation, simply remove this flag. To test a trained model, use the following command:

#### Test the trained low-level controller model
```
python calm/run.py
--test
--task HumanoidAMPGetup
--num_envs 16
--cfg_env calm/data/cfg/humanoid.yaml
--cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml
--motion_file [Your file path]/motions.yaml
--checkpoint [Your file path]/Humanoid_00014500.pth
```
&nbsp;

### High-level policy

#### High-level policy training
```
python calm/run.py
--task HumanoidSpecAnySKill
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--text_file calm/data/texts.yaml
--wandb_project_name special_policy
--render
```
`--llc_checkpoint` specifies the checkpoint to use for the low-level controller. `--text_file` specifies motion captions and their weights.
For both training method, we use pretrained model to extract the image features by default. If you want to render with camera, you just need add `--render` at the end.

#### Test the trained high-level model
```
python calm/run.py 
--test
--num_envs 16
--task HumanoidSpecAnySKill
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--render
--text_file calm/data/texts.yaml
--checkpoint [Your file path]/Humanoid_00000100.pth
```
`--checkpoint` here is the trained model with high-level policy.


#### Rigid scene policy training
```
python calm/run.py
--task HumanoidSpecAnySKillRigid
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--text_file calm/data/texts_rigid.yaml
--wandb_project_name special_policy_scene
--render
```
You can replace `--cfg_train` and `--text_file` with your own files.

#### Test the model trained with rigid object

```
python calm/run.py 
--test
--num_envs 16
--task HumanoidSpecAnySKillRigid
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--render
--text_file calm/data/texts_rigid.yaml
--checkpoint [Your file path]/Humanoid_00000050.pth
```


#### Articulated scene policy training
```
python calm/run.py
--task HumanoidSpecAnySKillArti
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/scene_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--text_file calm/data/texts_scene.yaml
--wandb_project_name special_policy_scene
--articulated
--render
```
Here we add `--articulated` to specify the articulated object in the scene.


#### Test the model trained with articulated object
```
python calm/run.py 
--test
--num_envs 16
--task HumanoidSpecAnySKillArti
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/scene_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--render
--articulated
--text_file calm/data/texts_scene.yaml
--checkpoint [Your file path]/Humanoid_00000100.pth
```
&nbsp;

[//]: # (### AMP)

[//]: # ()
[//]: # (We also provide an implementation of [Adversarial Motion Priors]&#40;https://xbpeng.github.io/projects/AMP/index.html&#41;.)

[//]: # (A model can be trained to imitate a given reference motion using the following command:)

[//]: # (```)

[//]: # (python calm/run.py --task HumanoidAMP --cfg_env calm/data/cfg/humanoid_sword_shield.yaml --cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml --motion_file calm/data/motions/reallusion_sword_shield/sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --headless  --track)

[//]: # (```)

[//]: # (The trained model can then be tested with:)

[//]: # (```)

[//]: # (python calm/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env calm/data/cfg/humanoid_sword_shield.yaml --cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml --motion_file calm/data/motions/reallusion_sword_shield/sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint [path_to_amp_checkpoint])

[//]: # (```)

[//]: # ()
[//]: # (&nbsp;)

### Motion Data

Motion clips are located in `calm/data/motions/`. Individual motion clips are stored as `.npy` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python calm/run.py
--test
--task HumanoidViewMotion
--num_envs 1
--cfg_env calm/data/cfg/humanoid.yaml
--cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml
--motion_file [Your file path].npy
```

`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.
If you want to retarget new motion clips to the character, you can take a look at an example retargeting script in `calm/poselib/retarget_motion.py`.


## Acknowledgments
Our code is based on [CALM](https://github.com/NVlabs/CALM) and [CLIP](https://github.com/openai/CLIP). Thanks for these great projects.

## Citation
```text
@inproceedings{cui2024anyskill,
  title={Anyskill: Learning Open-Vocabulary Physical Skill for Interactive Agents},
  author={Cui, Jieming and Liu, Tengyu and Liu, Nian and Yang, Yaodong and Zhu, Yixin and Huang, Siyuan},
  booktitle=Conference on Computer Vision and Pattern Recognition(CVPR),
  year={2024}
}
```

