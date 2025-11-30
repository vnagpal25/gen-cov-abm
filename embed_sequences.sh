#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=embed_sequences
#SBATCH --account=cse598s002f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=log.out
#SBATCH --error=error.out


# Set up environment

source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
conda activate gen_cov_abm

PROTEINS="n_sequence s_sequence"

# Optional: Override proteins from command line
if [ $# -gt 0 ]; then
    PROTEINS="$@"
fi

echo "Computing embeddings using ESM-2 for proteins: $PROTEINS"

# Run embedding generation with protein list
python embed-sequences.py $PROTEINS

echo "Embedding generation completed!"
