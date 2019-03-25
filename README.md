# CinCGAN-pytorch

This repository is a PyTorch version of the paper **"Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks"** from **CVPRW 2018**.


For super-resolution setting I refer to [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) (you can download pretrained EDSR from here)

This version is not good looking yet. 
It will be updated later..

--------------------------------------------------------------------------------------------

## Train

* Dataset 

  Download DIV2K dataset (NTIRE2018) from [here](https://competitions.codalab.org/competitions/18024#participate-get_data).
  800 training (~800) & 100 validation images (801~900)

* Pretrained EDSR network. 

  Download pretrained EDSR from [here](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar)

* Execution

  After move to 'code' folder, type the following command.
  	```bash
  	python main.py --dir_data 'data_path'
  	```
  data_path directory should contains 'DIV2K' dataset folder.  

## Test

* Execution

  After move to 'code' folder, type the following command.
  	```bash
  	python main.py --n_val 100 
  	```

