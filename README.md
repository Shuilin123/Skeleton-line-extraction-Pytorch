# ESPI
This is a uctrans-NET - based electronic speckle skeleton line extraction procedures 
## __Skeleton line extraction-Pytorch__ : _Electron scattering interference skeleton line extraction toolkit based on pytorch_
### Introduction
This project is a electron scattering interference skeleton line extraction code based on python and pytorch framework, including data preprocessing, model training and testing, visualization, etc. This project is suitable for researchers who study image segmentation.   
### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.7.9                    
pytorch                   1.7.0         
torchvision               0.8.0         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.3.2              
numpy                     1.19.2        
opencv                    3.4.2         
pandas                    1.1.3        
pillow                    8.0.1         
scikit-learn              0.23.2          
scipy                     1.5.2           
tensorboardX              2.1        
tqdm                      4.54.1             
```  
The above environment is successful when running the code of the project. In addition, it is well known that pytorch has very good compatibility (version>=1.0). Thus, __I suggest you try to use the existing pytorch environment firstly.__  

---  
## Usage 
### 0) Download Project 

Running```git clone https://github.com/Shuilin123/Skeleton line extraction-Pytorch.git```  
The project structure and intention are as follows : 
```
Skeleton line extraction-Pytorch			# Source code		
    ├── config.py		 	# Configuration information
    ├── lib			            # Function library
    │   ├── common.py
    │   ├── dataset.py		        # Dataset class to load training data
    │   ├── datasetV2.py		        # Dataset class to load training data with lower memory
    │   ├── extract_patches.py		# Extract training and test samples
    │   ├── help_functions.py		# 
    │   ├── __init__.py
    │   ├── logger.py 		        # To create log
    │   ├── losses
    │   ├── metrics.py		        # Evaluation metrics
    │   └── pre_processing.py		# Data preprocessing
    ├── models		        # All models are created in this folder
    │   ├── DenseUnet.py
    │   ├── __init__.py
    │   ├── LadderNet.py
    │   ├── nn
    │   └── UCTransNet.py
    ├── prepare_dataset	        # Prepare the dataset (organize the image path of the dataset)
    │   ├── speckle.py
    │   ├── data_path_list		  # image path of dataset
    │   
    ├── tools			     # some tools
    │   ├── ablation_plot.py
    │   ├── ablation_plot_with_detail.py
    │   ├── merge_k-flod_plot.py
    │   └── visualization
    ├── function.py			        # Creating dataloader, training and validation functions 
    ├── test.py			            # Test file
    └── train.py			          # Train file
```
### 1) Datasets preparation 
The dataset is currently not fully open source
3. Create data path index file(.txt). running:
Please modify the data folder path:`data_root_path`(in the [`speckle.py`](https://github.com/Shuilin123/Skeleton line extraction-Pytorch/master/prepare_dataset/drive.py) 
```
python ./prepare_dataset/speckle.py           
```
### 2) Training model
Please confirm the configuration information in the [`config.py`](https://github.com/Shuilin123/Skeleton line extraction-Pytorch/master/config.py). Pay special attention to the `train_data_path_list` and `test_data_path_list`. Then, running:
```
CUDA_VISIBLE_DEVICES=1 python train.py --save UCTransNet_Skeleton_line_extraction --batch_size 64
```
You can configure the training information in config, or modify the configuration parameters using the command line. The training results will be saved to the corresponding directory(save name) in the `experiments` folder.  
### 3) Testing model
The test process also needs to specify parameters in [`config.py`](https://github.com/Shuilin123/Skeleton line extraction-Pytorch/master/config.py). You can also modify the parameters through the command line, running:
```
CUDA_VISIBLE_DEVICES=1 python test.py --save UCTransNet_Skeleton_line_extraction  
```  
The above command loads the `best_model.pth` in `./experiments/UCTransNet_Skeleton_line_extraction` and performs a performance test on the testset, and its test results are saved in the same folder.    

## Visualization
0. Training sample visualization  


