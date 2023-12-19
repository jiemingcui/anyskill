# Anyskill for open-vocabulary physical natural motions

### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```

#### Low-level controller training

First, a CALM model can be trained to imitate a dataset of motions clips using the following command:
```
python calm/run_iter.py --task HumanoidAMPGetup --cfg_env calm/data/cfg/humanoid.yaml --cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml --motion_file ./motions/motions_155.yaml --headless  --track
```
`--motion_file` can be used to specify a dataset of motion clips that the model should imitate. 
The task `HumanoidAMPGetup` will train a model to imitate a dataset of motion clips and get up after falling.
Over the course of training, the latest checkpoint `Humanoid.pth` will be regularly saved to `output/`,
along with a Tensorboard log. `--headless` is used to disable visualizations and `--track` is used for tracking using weights and biases. If you want to view the
simulation, simply remove this flag. To test a trained model, use the following command:

##### Test the trained low-level controller model
```
python calm/run_iter.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env calm/data/cfg/humanoid\.yaml --cfg_train calm/data/cfg/train/rlg/calm_humanoid.yaml --motion_file ./motions/motions_155.yaml --checkpoint [path_to_calm_checkpoint]
```

&nbsp;

#### High-level policy

##### High-level policy training
```
python calm/run.py --task HumanoidSpecAnySKill --cfg_env calm/data/cfg/humanoid_anyskill.yaml --cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml --motion_file /home/cjm/CALM/calm/data/motions/target_height_motions/motions.yaml --llc_checkpoint /home/cjm/CALM/output/low-level/nn/Humanoid_00014500.pth --track --text_file calm/data/texts.yaml --wandb_project_name special_policy
```
`--llc_checkpoint` specifies the checkpoint to use for the low-level controller. `--text_file` specifies motion captions and their weights.
For both training method, we use pretrained model to extract the image features by default. If you want to render with camera, you just need add `--render` at the end.

##### Test the trained high-level model
```
python calm/run.py 
--test --num_envs 16 --task HumanoidSpecAnySKill
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--render
--text_file calm/data/texts.yaml
--checkpoint [Your file path]/Humanoid_00000600.pth
```

##### Rigid scene policy training
```
python calm/run.py --task HumanoidSpecAnySKillRigid --cfg_env calm/data/cfg/humanoid_anyskill.yaml --cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml --motion_file /home/cjm/CALM/calm/data/motions/target_height_motions/motions.yaml --llc_checkpoint /home/cjm/CALM/output/low-level/nn/Humanoid_00014500.pth --track --text_file calm/data/texts.yaml --render
```
##### Test the model trained with rigid object

```
python calm/run.py 
--test --num_envs 16 --task HumanoidSpecAnySKillRigid
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/spec_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--render
--text_file calm/data/texts_rigid.yaml
--checkpoint [Your file path]/Humanoid_00000600.pth
```


##### Articulated scene policy training
```
python calm/run.py --task HumanoidSpecAnySKillArti --cfg_env calm/data/cfg/humanoid_anyskill.yaml --cfg_train calm/data/cfg/train/rlg/scene_anyskill.yaml --motion_file /home/cjm/CALM/calm/data/motions/target_height_motions/motions.yaml --llc_checkpoint /home/cjm/CALM/output/low-level/nn/Humanoid_00014500.pth --track --text_file calm/data/texts_scene.yaml --wandb_project_name special_policy_scene --articulated --render
```
##### Test the model trained with articulated object
```
python calm/run.py 
--test --num_envs 16 --task HumanoidSpecAnySKillArti
--cfg_env calm/data/cfg/humanoid_anyskill.yaml
--cfg_train calm/data/cfg/train/rlg/scene_anyskill.yaml
--motion_file [Your file path]/motions.yaml
--llc_checkpoint [Your file path]/Humanoid_00014500.pth
--track
--render
--articulated
--text_file calm/data/texts_scene.yaml
--checkpoint [Your file path]/Humanoid_00000600.pth
```
&nbsp;

### AMP

We also provide an implementation of Adversarial Motion Priors (https://xbpeng.github.io/projects/AMP/index.html).
A model can be trained to imitate a given reference motion using the following command:
```
python calm/run.py --task HumanoidAMP --cfg_env calm/data/cfg/humanoid_sword_shield.yaml --cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml --motion_file calm/data/motions/reallusion_sword_shield/sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --headless  --track
```
The trained model can then be tested with:
```
python calm/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env calm/data/cfg/humanoid_sword_shield.yaml --cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml --motion_file calm/data/motions/reallusion_sword_shield/sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint [path_to_amp_checkpoint]
```

&nbsp;

### Motion Data

Motion clips are located in `calm/data/motions/`. Individual motion clips are stored as `.npy` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python calm/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env calm/data/cfg/humanoid_sword_shield.yaml --cfg_train calm/data/cfg/train/rlg/amp_humanoid.yaml --motion_file calm/data/motions/reallusion_sword_shield/sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy
```
`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.


If you want to retarget new motion clips to the character, you can take a look at an example retargeting script in `calm/poselib/retarget_motion.py`.