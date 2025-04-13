import os
import re

# Get all files in the current directory
files = os.listdir()

# Define the file patterns to match
file_patterns = ["output_test__", "output_rho_test__", "output_hydro_test__"]

# Iterate over all files
for file in files:
    for pattern in file_patterns:
        match = re.search(rf"({pattern})(\d+)_", file)
        if match:
            prefix = match.group(1)  # Get the file prefix (e.g., output_test__, output_rho_test__)
            num = int(match.group(2))   # Extract the number

            if 1 <= num <= 3000:
                new_num = num + 533  # Shift numbers (1 → 198, 2 → 199, ..., 47 → 244)

                # Create the new filename
                new_filename = re.sub(rf"{pattern}{num}_", f"{pattern}{new_num}_", file)

                # Rename (move) the file
                os.rename(file, new_filename)
                print(f"Renamed: {file} -> {new_filename}")

print("Renaming complete.")

