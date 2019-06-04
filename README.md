# RotationNet
fastai implementation of RotationNet

Read all about this project at: https://medium.com/@mathis.alex/rotationnet-in-fast-ai-adventures-in-virtual-set-creation-and-computer-vision-cd694ad7ec1b

<b>Contents:</b>

makeDataSubSet.py - creates a smaller dataset for quicker training and iteration.

rotationNet-fastai-MIRO.ipynb - jupyter notebook that trains and predicts using the MIRO dataset.

rotationNet-fastai.ipynb - jupyter notebook that trains and predicts using the ModelNet40 dataset.

rotationNet-fastai-expansion - added a new chair class to the MIRO dataset and uses resnet18 as the backbone. Best results: 96.07% class prediction accuracy, 77.32% pose estimate accuracy, 92.01% pose "precision" (pose estimate was only one rotation off from gt).

rNCallbacks.py - needed during inference.

startPredictors.py - script that needs to be running in the background when using from Unreal for inference.


See https://github.com/kanezaki/pytorch-rotationnet for the official PyTorch implementation of the paper and instructions on how to download and setup the MIRO and ModelNet datasets.

Note: Because of RotationNet's specific requirements I had to change a few lines in fast.ai's source code. I'm happy to provide details if interested.
