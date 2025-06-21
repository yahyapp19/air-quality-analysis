import pandas as pd
import os
import glob

# Define the path to the data directory
data_dir = '/data'
output_dir = '/temp_data'
output_file = os.path.join(output_dir, 'air_quality_combined.csv')

# Get a list of all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

# Check if files were found
if not csv_files:
    print(f"Error: No CSV files found in {data_dir}")
else:
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Initialize an empty list to store DataFrames
    all_df_list = []

    # Loop through each CSV file
    for f in csv_files:
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(f)
            
            # Extract station name from filename
            # Example filename: PRSA_Data_Aotizhongxin_20130301-20170228.csv
            station_name = os.path.basename(f).split('_')[2]
            
            # Add a 'station' column
            df['station'] = station_name
            
            # Append the DataFrame to the list
            all_df_list.append(df)
            print(f"Processed: {os.path.basename(f)}, Shape: {df.shape}")
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    if all_df_list:
        combined_df = pd.concat(all_df_list, ignore_index=True)
        
        # Display the shape and first few rows of the combined DataFrame
        print(f"\nCombined DataFrame Shape: {combined_df.shape}")
        print("\nCombined DataFrame Head:")
        print(combined_df.head())
        
        # Save the combined DataFrame to a new CSV file
        try:
            combined_df.to_csv(output_file, index=False)
            print(f"\nCombined data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving combined data: {e}")
    else:
        print("No dataframes were loaded, cannot combine.")

