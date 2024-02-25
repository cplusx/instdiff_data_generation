conda create -n datagen python=3.8 -y
source activate datagen
conda install nvidia/label/cuda-12.2.1::cuda -y

pip install torch torchvision torchaudio
pip install git+https://github.com/facebookresearch/segment-anything.git
# pip install -U openmim
# pip install psutil ninja # Addresses the slow installation issue of mmcv
# mim install mmcv
git clone https://github.com/IDEA-Research/GroundingDINO.git
python -m pip install -e GroundingDINO
pip install -U setuptools # important for RAM, or you will get error
pip install git+https://github.com/xinyu1205/recognize-anything.git

pip install -U diffusers
pip install -U transformers
pip install einops transformers_stream_generator
pip install tiktoken

# if you have not retrieved the checkpoints yet, uncomment the following lines
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
# wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth