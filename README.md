## Setup
    ```
    conda create -n sdc1 python==3.6.8
    conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
    pip install -r requirements.txt
    pip install cmake
    pip install opencv-python==4.5.3.56
    cd pointnet2
    python setup.py install
    cd ..
    pip install -r requirements.txt
    pip install future
    ```
## Version
- easydict==1.10
- matplotlib==3.3.4
- numpy==1.19.2
- opencv_python==4.5.3.56
- Pillow==10.0.0
- plyfile==0.8
- scipy==1.5.4
- setuptools==58.0.4
- torch==1.1.0
- trimesh==2.35.39
