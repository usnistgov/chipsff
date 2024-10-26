#!/bin/bash
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --partition=rack1,rack2,rack2e,rack3,rack4,rack4e,rack5,rack6
#SBATCH --job-name=jid_calculator_array
#SBATCH --output=logs/jid_calculator_%A_%a.out
#SBATCH --error=logs/jid_calculator_%A_%a.err

source activate umlff-final

# Define arrays of JIDs and calculators
jid_list=('JVASP-1002' 'JVASP-816' 'JVASP-867' 'JVASP-1029' 'JVASP-861' 'JVASP-30')
calculator_types=("chgnet" "mace")

# Calculate the total number of combinations
total_jids=${#jid_list[@]}
total_calculators=${#calculator_types[@]}
total_jobs=$((total_jids * total_calculators))

# Get the current task ID
task_id=$SLURM_ARRAY_TASK_ID

# Calculate indices for JID and calculator
jid_index=$((task_id / total_calculators))
calculator_index=$((task_id % total_calculators))

# Check if indices are within bounds
if [ $jid_index -lt $total_jids ] && [ $calculator_index -lt $total_calculators ]; then
  jid=${jid_list[$jid_index]}
  calculator=${calculator_types[$calculator_index]}

  echo "Processing JID: $jid with calculator: $calculator"

  # Generate input.json for this task
  input_file="input_${jid}_${calculator}.json"
  cat > $input_file <<EOL
{
  "jid": "$jid",
  "calculator_type": "$calculator",
  "chemical_potentials_file": "chemical_potentials.json",
  "properties_to_calculate": [
    "relax_structure",
    "calculate_ev_curve",
    "calculate_formation_energy",
    "calculate_elastic_tensor",
    "run_phonon_analysis",
    "analyze_surfaces"
  ],
  "relaxation_settings": {
    "fmax": 0.05,
    "steps": 200
  },
  "phonon_settings": {
    "dim": [1, 1, 1],
    "distance": 0.2
  },
  "surface_indices_list": [
    [0, 1, 0]
  ]
}
EOL

  # Run the analysis
  python run_chipsff.py --input_file $input_file
else
  echo "Task ID $task_id is out of range."
fi
