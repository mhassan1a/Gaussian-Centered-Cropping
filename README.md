# Research and Development Project
## Title: The Effect of the Randomness Underlying Image Cropping in Contrastive Learning: A Comparative Study 
* Abstract—In the contemporary landscape of computer vision and deep learning (DL), contrastive learning (CL) stands out as one of the most important self-supervised learning (SSL) frameworks. CL offers the advantage of learning directly from unlabeled data by leveraging fundamental knowledge representation principles, thereby enabling DL models to learn features that are transferable to downstream tasks. However, the success of CL methods is highly dependent on the effective use of robust image augmentation techniques, particularly image cropping. CL methods utilize randomized cropping (RC) to produce semantically related views (i.e. positive pairs) that serve as self-labels. However, RC can also introduce false positives, where views from different classes are incorrectly treated as positive pairs, significantly degrading performance.

* Consequently, the primary challenge associated with image cropping, in the context of CL, lies in striking a balance between introducing non-trivial positive pairs and minimizing the occurrence of false positives. To address this challenge, this project proposes a novel approach: A parameterized cropping method, Gaussian-Centered Cropping (GCC), that facilitates the fine-tuning of the cropping process to reduce the likelihood of false positives and improve the performance. The experimental results demonstrate that the proposed method outperforms Random Crop by 2.7, 6.7, 9.5, and 12.4 percentage points for crop sizes of 20\%, 40\%, 60\%, and 80\% respectively, at the same computational cost. In addition, an enhanced version of GCC, Multi-object Gaussian-Centered Cropping (MGCC), is presented to handle images containing multiple objects.
#
This project has been submitted by `Mohamed Hassan` to the Department of Computer Science at Hochschule Bonn-Rhein-Sieg in partial fulfillment of the requirements for the degree of Master of Science in Autonomous Systems.


Supervised by 
- Prof. Dr.-Ing. Sebastian Houben 
- Mohammad Wasil, M.Sc.

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

To use this project, follow the steps below:

1. Clone the repository to your local machine by running the following command in your terminal:

```bash
git clone https://github.com/mhassan1a/Research-and-Development-Project-Msc-.git
```

2. Navigate to the project directory:

```bash
cd Research-and-Development-Project-Msc
```

3. Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

4. Run the training script with a single GPU using the following command:

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

5. If you have multiple GPUs, you can run the training script with multiple GPUs using the following command:

```bash
python ./src/train_ngpus.py \
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





