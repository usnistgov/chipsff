#!/bin/bash

interface_data=(
  "JVASP-816 JVASP-32 1_1_1 0_0_1"
  "JVASP-816 JVASP-39 1_1_1 0_0_1"
  "JVASP-39 JVASP-32 1_0_0 0_0_1"
  "JVASP-664 JVASP-32 0_0_1 0_0_1"
  "JVASP-867 JVASP-14741 1_1_1 1_1_1"
  "JVASP-867 JVASP-79561 1_1_1 0_0_1"
  "JVASP-816 JVASP-14741 1_1_1 1_1_1"
  "JVASP-816 JVASP-79561 1_1_1 0_0_1"
  "JVASP-867 JVASP-58349 1_1_1 1_0_1"
  "JVASP-816 JVASP-58349 1_1_1 1_0_1"
  "JVASP-664 JVASP-652 0_0_1 0_0_1"
  "JVASP-667 JVASP-688 0_0_1 0_0_1"
  "JVASP-667 JVASP-58349 0_0_1 1_0_1"
  "JVASP-867 JVASP-58349 1_1_1 0_0_1"
  "JVASP-867 JVASP-1216 1_0_0 1_1_1"
  "JVASP-1029 JVASP-14741 0_0_1 0_0_1"
  "JVASP-867 JVASP-7809 0_0_1 1_1_0"
  "JVASP-867 JVASP-667 1_1_1 0_0_1"
  "JVASP-867 JVASP-48 1_1_1 0_0_1"
  "JVASP-867 JVASP-1216 1_1_1 1_1_1"
  "JVASP-5 JVASP-14741 1_1_0 0_0_1"
  "JVASP-104 JVASP-14741 0_0_1 1_1_1"
  "JVASP-104 JVASP-14741 0_0_1 0_0_1"
)

# 2) Define one or more calculators:
calculators=("mattersim")
# If you have more calculators, e.g. ("chgnet" "vasp"), you can add them here.

# 3) Loop over interfaces & calculators, generate sub-scripts & submit
for item in "${interface_data[@]}"; do
  
  # Split string into variables
  read -r film_jid substrate_jid film_index substrate_index <<< "$item"
  
  for calc in "${calculators[@]}"; do
    
    # Construct a unique name for the job & sub-script
    job_name="interface_${film_jid}_${substrate_jid}_${film_index}_${substrate_index}_${calc}"
    sub_script="submit_${job_name}.sh"
    
    echo "Creating and submitting job: $job_name"

    # --------------------------------------------------------------------------
    # Write out the sub-script that SLURM will run
    # --------------------------------------------------------------------------
    cat <<EOF > $sub_script
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=30-00:00:00
#SBATCH --partition=rack1,rack2e,rack3,rack4,rack4e,rack5,rack6
#SBATCH --job-name=$job_name
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err

# Module loads or conda activation if needed
# module load python
# conda activate myenv

# Generate a JSON input file
cat > input_${job_name}.json <<JSON
{
  "film_jid": ["$film_jid"],
  "substrate_jid": ["$substrate_jid"],
  "calculator_type": "$calc",
  "chemical_potentials_file": "chemical_potentials.json",
  "film_index": "$film_index",
  "substrate_index": "$substrate_index",
  "properties_to_calculate": [
    "analyze_interfaces"
  ]
}
JSON

# Run the job with your Python script
python run_chipsff.py --input_file input_${job_name}.json
EOF
    # --------------------------------------------------------------------------
    
    # Make sub-script executable
    chmod +x $sub_script

    # Submit the sub-script to SLURM
    sbatch $sub_script

  done
done
