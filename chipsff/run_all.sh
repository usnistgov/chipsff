#!/bin/bash

# Define your list of JIDs and calculator types

jid_list=('JVASP-1002' 'JVASP-816' 'JVASP-867' 'JVASP-1029' 'JVASP-861''JVASP-30' 'JVASP-8169' 'JVASP-890' 'JVASP-8158''JVASP-8118'
    'JVASP-107' 'JVASP-39' 'JVASP-7844' 'JVASP-35106' 'JVASP-1174'
    'JVASP-1372' 'JVASP-91' 'JVASP-1186' 'JVASP-1408' 'JVASP-105410'
    'JVASP-1177' 'JVASP-79204' 'JVASP-1393' 'JVASP-1312' 'JVASP-1327'
    'JVASP-1183' 'JVASP-1192' 'JVASP-8003' 'JVASP-96' 'JVASP-1198'
    'JVASP-1195' 'JVASP-9147' 'JVASP-41' 'JVASP-34674' 'JVASP-113'
    'JVASP-32' 'JVASP-840' 'JVASP-21195' 'JVASP-981' 'JVASP-969'
    'JVASP-802' 'JVASP-943' 'JVASP-14812' 'JVASP-984' 'JVASP-972'
    'JVASP-958' 'JVASP-901' 'JVASP-1702' 'JVASP-931' 'JVASP-963'
    'JVASP-95' 'JVASP-1201' 'JVASP-14837' 'JVASP-825' 'JVASP-966'
    'JVASP-993' 'JVASP-23' 'JVASP-828' 'JVASP-1189' 'JVASP-810'
    'JVASP-7630' 'JVASP-819' 'JVASP-1180' 'JVASP-837' 'JVASP-919'
    'JVASP-7762' 'JVASP-934' 'JVASP-858' 'JVASP-895'
)
calculator_types=("chgnet","mace")

# Loop over each combination of JID and calculator type
for jid in "${jid_list[@]}"; do
  for calculator in "${calculator_types[@]}"; do

    # Submit the job to SLURM using sbatch with the --wrap option
    sbatch <<EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --partition=rack1,rack2,rack2e,rack3,rack4,rack4e,rack5,rack6
#SBATCH --job-name=${jid}_${calculator}
#SBATCH --output=logs/${jid}_${calculator}_%j.out
#SBATCH --error=logs/${jid}_${calculator}_%j.err



# Run your Python script with the necessary arguments
python master.py --jid $jid --calculator_type $calculator --chemical_potentials_json "\$(cat chemical_potentials.json)"
EOT

  done
done

