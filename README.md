# Research and Development Project
## Title: The Effect of the Randomness Underlying Image Cropping in Contrastive Learning: A Comparative Study 
* Abstract—In the contemporary computer vision and deep
learning landscape, contrastive learning (CL) stands out as
one of the most important self-supervised learning (SSL)
frameworks. CL offers the advantage of learning directly
from unlabeled data by leveraging fundamental knowledge
representation principles, thereby enabling the acquisition of
meaningful representations that are transferable to downstream
tasks. However, the success of CL methods is highly dependent
on the effective use of robust image augmentation techniques,
particularly random cropping.
CL methods utilize random cropping to produce similar and
semantically related views, termed positive pairs. By contrasting
these views with one another, these methods aim to learn and
capture representative features of the classes. However, random
cropping can also introduce false positives, where views from
different classes are incorrectly treated as positive pairs, leading
to a significant degradation in performance. Consequently, the
primary challenge associated with the stochastic generation of
crops lies in striking a balance between introducing non-trivial
positive pairs and minimizing the occurrence of false positives.
To tackle this challenge, this project proposes a novel ap-
proach: a parameterized cropping method, Gaussian Centered
Cropping (GCC), that facilitates hyper-tuning of the cropping
process, thereby reducing the occurrence of false positives. The
experimental results demonstrate that the proposed method
outperforms traditional random cropping by 2.7, 6.7, 9.5, and
12.4 percentage points for crop sizes of 20%, 40%, 60%, and
80% respectively, at the same computational cost. In addition,
an enhanced version of GCC, Multi-object Gaussian Centered
Cropping (MGCC), is presented to handle images containing
multiple objects, further reducing the likelihood of false positives.
Index Terms—Contrastive Learning, Random Cropping, Ran-
domness, Data Augmentation, Parametric Random Cropping,
Computer Vision.

###### Submitted to the Department of Computer Science at Hochschule Bonn-Rhein-Sieg in partial fulfillment of the requirements for the degree of Master of Science in Autonomous Systems.

## Folder Hierarchy

The folder hierarchy for the project is as follows:

```
rnd_cluster/
├── README.md
├── src/
│   ├── train_1gpu.py
│   ├── train_ngpus.py
│   ├── cropping.py
│   ├── models.py
│   ├── visualization.ipynb
│   └── ...
├── requirements.txt
└── ...
```

## File Descriptions

Here is a brief description of the files in the project:

- `train_1gpu.py`: This file contains the code for training the model using a single GPU.
- `train_ngpus.py`: This file contains the code for training the model using multiple GPUs.
- `cropping.py`: This file implements the parameterized cropping method, Gaussian Centered Cropping (GCC), and Multi-object Gaussian Centered Cropping (MGCC) as well as the their corrected versions.
- `models.py`: This file contains the implementation of the models used in the project.
- `visualization.ipynb`: This Jupyter notebook provides visualization tools for analyzing the trained models.
- `requirements.txt`: This file lists all the required dependencies for the project.

Feel free to explore these files to understand the project structure and functionality.






# Usage
To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

For training with one gpu, you can use the following command:

```bash
python ./src/train_1gpu.py \
    --method gcc \
    --crop_size 0.2 \
    --std  3 \
    --num_of_trials 1 \
    --pretrain_epoch 200 \
    --num_workers 3 \
    --hidden_dim 128 \
    --batchsize 512 \
    --clf_epochs 100 \
    --dataset 'Cifar10' \
    --model 'Proto18' \
    --adaptive_center True\
    --job_id 0\
    --min_max 0.25 0.75
```

for Multi-GPUs change `train_1gpu` to `train_ngpus`

Note: Please make sure you have a GPU available for training the model. The code provided assumes that you have access to a GPU for faster computation. If you don't have a GPU, you may experience slower training times or may need to modify the code to run on a CPU-only setup.



## Submission

This project has been submitted to the Department of Computer Science at Hochschule Bonn-Rhein-Sieg in partial fulfillment of the requirements for the degree of Master of Science in Autonomous Systems.

By Mohamed Hassan

And

Supervised by Prof. Dr.-Ing. Sebastian Houben and Mohammad Wasil, M.Sc.

