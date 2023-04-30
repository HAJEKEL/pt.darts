# DARTS: Differentiable Architecture Search

Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." arXiv preprint arXiv:1806.09055 (2018). [[arxiv](https://arxiv.org/abs/1806.09055)]

# Original verion of this repository

Refer to [khanrc](https://github.com/khanrc/pt.darts) to get this repository without my contribution. However, note that I was unable to perform a DARTS search with that repository out of the box. You would likely have to resolve some bugs.

# My contribution

The aim of this repository is to provide a DARTS software that anyone with little DARTS expertise can use. In my experience there is no othere repository available on github that provides that. This repository contains the DARTS software and this README elaborates on how to change the operations that can be used within the DARTS cells. To get more information on how to run a search, please refer to the master branch of my [EfficientNetV2_paper_reproduction](https://github.com/HAJEKEL/EfficientNetV2_paper_reproduction) repo. 

## Operations

The [ops.py](https://github.com/HAJEKEL/pt.darts/blob/master/models/ops.py) file contains the operations out of which the DARTS search algorithm can choose. In there, I defined 3 new blocks:
-   SepConv: This is the depthwise seperable convolution from the Mobilenetv1 paper.
-   MBConv: This is the main building block from the EfficientnetV1 paper.
-   FusedMBConv: This is the main building block from the EfficientnetV2 paper. 

I kept the original blocks defined by [khanrc](https://github.com/khanrc/pt.darts) such that you will have a range of blocks to choose from out of the box. These orignial blocks include average/max pooling, dilated convolution and more. 

Define the search:
-   Insert the operations you DARTS to choose from in the [OPS](https://github.com/HAJEKEL/pt.darts/blob/33a336d8c3e1f785de583a480e9ea6aa8e0bd181/models/ops.py#L7) dictionary. If you're operations are not defined yet, you should define them. 
-   Inside [genotypes.py](https://github.com/HAJEKEL/pt.darts/blob/master/genotypes.py) you should change the PRIMITVES list accordingly. 

## GIF of search results

I added a [GIF maker](https://github.com/HAJEKEL/pt.darts/blob/master/gif_maker.py) that allows you to make a GIF out of the search results. 

## Loss/accuracy plot

I added a [plotter](https://github.com/HAJEKEL/pt.darts/blob/master/plot_train_val.py) that generates a plot with loss and accuracy progression throughout the search for both training and validation data. 