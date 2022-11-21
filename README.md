<!-- Disclaimer: This GitHub repository is under routine maintenance. -->
# StyleFool

This is the source code for our SP'23 paper "StyleFool: Fooling Video Classification Systems via Style Transfer", partly based on style transfer code by Manuel Ruder https://github.com/manuelruder/artistic-videos.

Our algorithm uses style transfer for video black-box attacks, and ensures the indistinguishability and consistency of the adversarial video.


## Setup

Tested with Ubuntu 18.04.

* Please follow https://github.com/manuelruder/artistic-videos to install torch7, loadcaffe, DeepFlow and the CUDA backend.
* Run `bash models/download_models.sh` to download the VGG model.
* Please install ffmpeg for convertion between videos and images.


## Requirements

- GPU is required in this project. If you run in CPU, the speed will be maddeningly slow.
- Python==3.7
- pytorch==1.4.0
- torchvision==0.5.0
- numpy==1.16.2


## Dataset

* You need to download the action recognition dataset UCF-101 (http://crcv.ucf.edu/data/UCF101.php), and save it in 'dataset/'. Copy the folder and rename it as 'UCF-101_images'.
* cd dataset and run `bash ./list/convert_video_to_images.sh ./UCF-101_images 5` to decode the video into images.
* Copy the 'UCF-101_images' folder and rename it as 'UCF-101_npy'.
* Run `python png2npy.py` to generate npy files for the dataset.


## Pretrained model
We use the pre-trained C3D model from [here](https://1drv.ms/u/s!Aj2hSJitqRWpeT96f1QG1UbKVhA).


## Usage

### Attack preparation

**Targeted attack**

Run `python prepare.py --model C3D --dataset UCF101 --target`.

**Untargeted attack**

Run `python prepare.py --model C3D --dataset UCF101`.

The processes of generating styles, calculating color themes and calculating target class confidence are needed only once. If you have finished those processes, run `python prepare.py --model C3D --dataset UCF101 --generate-styles --color --superposition --target` for targeted attack preparation and `python prepare.py --model C3D --dataset UCF101 --generate-styles --color --superposition` for untargeted attack preparation.

Then, you will find 'batch_command_target.sh' and 'batch_command_untarget.sh' in your directory.

**Basic arguments**:
* `--model`: The attacked model. Default: C3D
* `--dataset`: The dataset. Default: UCF101
* `--generate-styles`: Need to generate styles. Default: True
* `--color`: Need to calculate the color themes. Default: True
* `--superposition`: Need to calculate the target class confidence. Default: True
* `--style-folder`: Path to save all the style images.
* `--prepare-save-path`: Path to save color themes and target class confidence.
* `--gpu`: ID of the GPU to use; for CPU mode set `-gpu` to -1, but CPU is not recommended.
* `--dataset-img-path`: The path of dataset in png forms.
* `--dataset-video-path`: The path of dataset in avi forms. 
* `--attack-id`: Attack_id. You can change it to other numbers.
* `--random-seed`: Random_seed when choosing target label. You can change it to other numbers.
* `--target`: Target attack or untarget attack (default). Default: False


### Style transfer

**Targeted attack**

Run `sh ./batch_command_target.sh`.

**Untargeted attack**

Run `sh ./batch_command_untarget.sh`.

### Adversarial attack preparation

**Targeted attack**

Run `python attack_prepare.py --model C3D --dataset UCF101 --target`.

**Untargeted attack**

Run `python attack_prepare.py --model C3D --dataset UCF101`.

**Basic arguments**:
* `--model`: The attacked model. Default: C3D
* `--dataset`: The dataset. Default: UCF101
* `--dataset-npy-path`: The path of dataset in npy forms.
* `--gpu`: ID of the GPU to use; for CPU mode set `-gpu` to -1, but CPU is not recommended.
* `--styled-npy-path`: The path of styled_npy.
* `--output-npy-path`: The path of output adversarial_npy.
* `--target`: Target attack or untarget attack (default). Default: False

Then, you will find 'target_attack_batch.sh' and 'untarget_attack_batch.sh' in your directory.

### Adversarial attack

**Targeted attack**

Run `sh ./target_attack_batch.sh`.

**Untargeted attack**

Run `sh ./untarget_attack_batch.sh`.

### Generate videos (optional)

If you want to visualize the adversarial video and the initial video, run `python generate_videos.py`. 

**Basic arguments**:
* `--output-npy-path`: The path of output adversarial_npy.
* `--output-video-path`: The path of output adversarial videos.


## Acknowledgement
* Part of our implementation is based on Manuel Ruder's [artistic-videos](https://github.com/manuelruder/artistic-videos) and Linxi Jiang's [VBAD](https://github.com/Jack-lx-jiang/VBAD).

## Citation

If you use this code or its parts in your research, please cite the following paper:

```
@inproceedings{cao2023stylefool,
  title={StyleFool: Fooling Video Classification Systems via Style Transfer},
  author={Cao, Yuxin and Xiao, Xi and Sun, Ruoxi and Wang, Derui and Xue, Minhui and Wen, Sheng},
  booktitle={2023 IEEE Symposium on Security and Privacy (SP)},
  year={2023},
  organization={IEEE},
  address={San Francisco, CA, USA},
  month={May}
}
```
