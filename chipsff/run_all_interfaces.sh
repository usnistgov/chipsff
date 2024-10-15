#!/bin/bash

# Define your list of JIDs for films and substrates and calculator types

film_jid_list=('JVASP-1002' 'JVASP-816' 'JVASP-867' 'JVASP-1029')  # Add your actual film JIDs
substrate_jid_list=('JVASP-107' 'JVASP-39' 'JVASP-7844' 'JVASP-35106')  # Add your actual substrate JIDs

calculator_types=("chgnet" "mace")

film_index="1_1_0"  # Adjust Miller indices as needed
substrate_index="1_1_0"  # Adjust Miller indices as needed

# Loop over each combination of film JID, substrate JID, and calculator type
for film_jid in "${film_jid_list[@]}"; do
  for substrate_jid in "${substrate_jid_list[@]}"; do
    for calculator in "${calculator_types[@]}"; do

      # Submit the job to SLURM using sbatch with the --wrap option
      sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --partition=rack1
#SBATCH --job-name=${film_jid}_${substrate_jid}_${calculator}
#SBATCH --output=logs/${film_jid}_${substrate_jid}_${calculator}_%j.out
#SBATCH --error=logs/${film_jid}_${substrate_jid}_${calculator}_%j.err

# Run your Python script with the necessary arguments for interfaces
python master.py --film_jid $film_jid --substrate_jid $substrate_jid --calculator_type $calculator --film_index $film_index --substrate_index $substrate_index 
EOT

    done
  done
done

