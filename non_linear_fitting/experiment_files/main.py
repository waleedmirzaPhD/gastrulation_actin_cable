import pandas as pd

# Function to modify the first column by adding 0.2
def modify_first_column(input_file, output_file):
    # Read the file
    df = pd.read_csv(input_file, header=None, delimiter=',', dtype=float)
    
    # Add 0.2 to the first column
    df[0] += 0.2
    
    # Save the modified data to a new file
    df.to_csv(output_file, header=False, index=False, float_format='%.8E')
    
    print(f"Modified file saved as: {output_file}")

# Example usage
input_filename = "normal_temp2.txt"  # Change to your actual input file
output_filename = "normal_temp3.txt" # Change to your desired output file
modify_first_column(input_filename, output_filename)

