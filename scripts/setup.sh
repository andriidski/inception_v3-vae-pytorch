mkdir inception-vae
cd inception-vae
mkdir slurm-logs
module purge
module load pytorch/python3.6/gnu/20171124
python3 -m venv env
source env/bin/activate
pip3 install torch torchvision