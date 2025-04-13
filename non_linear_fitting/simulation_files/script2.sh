#!/bin/bash

# Enable debug mode to print each command before executing it
set -x

# Define variables for remote connection
REMOTE_HOST="mirza@login1.cluster.embl.de"
REMOTE_DIR="/home/mirza/run/wrong_paper/final/"
LOCAL_DIR=$(pwd) # Current directory
PASSWORD="determinedE123$" # Password

# Save the current directory
current_dir=$(pwd)

# Loop through i from 1 to 125 to construct folder names
for i in $(seq 971 2500); do
  # Define the folder name pattern
  folder_name="test__${i}_param1_*"
 
  # Find matching folders on the remote server
  folder_list=$(sshpass -p "$PASSWORD" ssh "$REMOTE_HOST" "ls -d ${REMOTE_DIR}/${folder_name} 2>/dev/null")

  # Loop through all matching folders and download the output.txt file
  for remote_folder in $folder_list; do
    # Extract the folder name from the remote folder path
    folder_name=$(basename "$remote_folder")
    
    
    # Define the remote output.txt file path
    remote_file_path="${remote_folder}/output.txt"
    remote_file_path_rho="${remote_folder}/output_rho.txt"
    remote_file_path_hydro="${remote_folder}/output_hydro.txt"



    local_file_path="${current_dir}/output_${folder_name}.txt"
    local_file_path_rho="${current_dir}/output_rho_${folder_name}.txt"    
    local_file_path_hydro="${current_dir}/output_hydro_${folder_name}.txt"



    # Use sshpass and rsync to transfer the output.txt file
    echo "Downloading output.txt for directory ${folder_name}..."
    sshpass -p "$PASSWORD" rsync -avzh "${REMOTE_HOST}:${remote_file_path}"     "$local_file_path"
    sshpass -p "$PASSWORD" rsync -avzh "${REMOTE_HOST}:${remote_file_path_rho}" "$local_file_path_rho"
    sshpass -p "$PASSWORD" rsync -avzh "${REMOTE_HOST}:${remote_file_path_hydro}"  "$local_file_path_hydro"



    if [ -f "$local_file_path" ]; then
      echo "output.txt downloaded successfully for directory ${folder_name}."
    else
      echo "output.txt not found in ${folder_name} or failed to download."
    fi
  done
done

# Return to the original directory
cd "$current_dir" || exit

# Print completion message
echo "Download of output.txt files complete."

