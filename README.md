# ASDPlatform

   ## Installation

   __My Computing Env__
    
    - OS: CentOS7
    - gcc: 8.3.1
    - g++: 8.3.1
    - nvidia: 12.1
    - cuda: 11.6
    
   __create env__
    
    1. conda create --name "name" python=3.8.13 -y
    
    2. conda activate "name"
    
      
    
  __install library__
    
    1. git clone https://github.com/open-mmlab/mmaction2.git
    
    2. cd mmaction2
    
    3. pip install torchvision
    
    4. pip install -U openmim
    
    5. pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
    (depending on your computing env)
    
    6. mim install mmengine
    
    7. mim install mmdet
    
    8. pip install scikit-learn
    
    9. pip install seaborn
    
    10. pip install wandb
