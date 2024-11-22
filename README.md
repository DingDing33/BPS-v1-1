
# BPS-v1-1

# Baysian Power Steering

This is the official release of  [Bayesian Power Steering: An Effective Approach for Domain Adaptation of Diffusion Models](https://arxiv.org/pdf/2406.03683).

Bayesian Power Steering(BPS) is a neural network structure to guiding the Stable Diffusion(SD) towards directions with a high probability density of satisfying the given condition.

![img](github_docs/imgs/ps.jgg)

BPS employs a top-heavy integration structure to achieve differentiated control across different ports. With a lightweight CNN network that excludes memory-intensive cross-attention structures, it is capable of performing fine-tuning tasks suitable for small-scale tasks. In the diagram above, the blue box represents the BPS architecture, with darker regions indicating more complex structures and higher control weights.

# Stable Diffusion + Baysian Power Steering

Using the BPS, we can guide SD in this way:

![img](github_docs/imgs/bps.jgg)

We adapt standard SD1.5. It can be downloaded from the [official page of Stability](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and the file is "v1-5-pruned.ckpt".


# Environment
```console
conda create -n bps python=3.8.18
conda env update -n bps --file environment.yaml
```

# Documents

For training the BPS, the .py files **train.py** and **train_dataset.py** is of interest. And they mainly primarily calls the folder **ldm**, which contains the classes and functions needed for the SD model, and **cldm**, which contains the relevant code needed for BPS.

Folder **models** contains the downloaded pre-trained SD models and the yaml file storing the hyperparameters of BPS.

The preprocessed data will be placed in folder **data** and the code related to preprocessing can be found in folder **annotator**.

# train a Baysian Power Steering to guide the SD system

We are now entering the highly anticipated phase, where we are about to fulfill your goal of training your very own BPS! Perhaps you have an idea for your perfect research project and would like to control the generation of images within a specific range using your own methods to expand your research dataset. You may have already manually or automatically annotated some data for this purpose. Here, the controls can be anything that can be transformed into an image, such as edges, layers, or even styles! We are pleased to inform you that even with just 40 annotated data points, BPS is more than capable of handling the task!

We hope that after reading this section, you will find training BPS to be incredibly simple.

## Design conditions + data preprocess
Design the conditions according to your target domain. And preprocess the image data to get the matched conditions. The conditions and images need to be placed in the **source and target folder** under foder **data**, and **a json file** is needed to record the pairing relationship between the conditions and images.

Finally, add a Datase class to **train_dataset.py** according to your own data format.

## Setting the hyperparameters of the model
Create a yaml file under folder **models**.

## Train
```console
python train.py
```



