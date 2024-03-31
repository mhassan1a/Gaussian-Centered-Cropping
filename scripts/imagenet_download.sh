#!/bin/bash
#SBATCH --partition=any         # partition / wait queue
#SBATCH --nodes=1               # number of nodes
#SBATCH --tasks-per-node=1      # number of tasks per node
#SBATCH --time=3-0:00:00        # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --mem=1G                # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --output=./out/train_net-%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=./out/train_net-%j.err     # filename for STDERR

# Print a message indicating the start of the job
echo "Starting Job"

# Print the job ID allocated by Slurm
echo "Job ID: $SLURM_JOB_ID"

# Define the directory to store ImageNet data
imagenet_dir="/work/mhassa2s/data/imagenet"

# Check if the directory exists, create it if not
if [ ! -d "$imagenet_dir" ]; then
    mkdir -p "$imagenet_dir"
fi

# Function to download files with error handling
download_file() {
    url=$1
    filename=$2
    echo "Downloading $filename from $url"
    wget -q --show-progress "$url" -P "$imagenet_dir" || { echo "Error downloading $filename"; exit 1; }
}

# Download ImageNet data files
download_file "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" "ILSVRC2012_img_train.tar"
download_file "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test.tar" "ILSVRC2012_img_test.tar"
download_file "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar" "ILSVRC2012_img_test_v10102019.tar"

# Print a message indicating successful completion
echo "Job completed successfully"
