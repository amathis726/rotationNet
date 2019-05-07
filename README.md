# RotationNet
fastai implementation of RotationNet

Read all about this project at: https://medium.com/@mathis.alex/rotationnet-in-fast-ai-adventures-in-virtual-set-creation-and-computer-vision-cd694ad7ec1b

Contents:
makeDataSubSet.py - creates a smaller dataset for quicker training and iteration.
render_shaded_black_bg.blend - blender script for creating your own datasets.
rotationNet-fastai-MIRO.ipynb - jupyter notebook that trains and predicts using the MIRO dataset.
rotationNet-fastai.ipynb - jupyter notebook that trains and predicts using the ModelNet40 dataset.

See https://github.com/kanezaki/pytorch-rotationnet for instructions on how to download and setup the MIRO and ModelNet datasets.

Note: Because of RotationNet's specific requirements I had to change a few lines in fast.ai's source code. I'm happy to provide details if interested.
