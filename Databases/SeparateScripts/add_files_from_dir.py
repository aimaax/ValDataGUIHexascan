import os
import pandas as pd
from datetime import datetime

# Define root path
# ROOT_PATH_IMAGES = r"E:\CERN\BackupJuly2024\outputs"
ROOT_PATH_IMAGES = r"E:\CERN\BackupJuly2024\Outputs_Lab_PC"

# Load database
# path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "updated_validation_DB_new")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB_Lab_PC", "lab_PC_DB")
DB = pd.read_pickle(path)

# # Ensure the correct indices are set
# DB.set_index(["Campaign", "DUT", "FileName", "bID"], inplace=True)

# Extract existing (Campaign, DUT, FileName) combinations from DB
existing_combinations = set(DB.index.to_frame()[['Campaign', 'DUT', 'FileName']].itertuples(index=False, name=None))

# List to store missing files information
missing_files = []

def check_and_add_files(root_path):
    for campaign in os.listdir(root_path):
        campaign_path = os.path.join(root_path, campaign)
        if not os.path.isdir(campaign_path):
            continue
        
        for dut in os.listdir(campaign_path):
            dut_path = os.path.join(campaign_path, dut)
            if not os.path.isdir(dut_path):
                continue
            
            for file in os.listdir(dut_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(dut_path, file)
                    # Check if the file size is larger than 100KB
                    if os.path.getsize(file_path) < 100000:  # Less than 100KB
                        print(f"File {file_path} is too small and will be deleted.")
                        os.remove(file_path)  # Delete the file
                        continue
                    
                    file_name = file[:-4]  # Remove .npy extension
                    
                    # Check if the combination of Campaign, DUT, and FileName already exists in the database
                    combination = (campaign, dut, file_name)
                    if combination not in existing_combinations:
                        print(campaign, dut, file_name)
                        new_row_index = (campaign, dut, file_name, "nan")
                        new_row_data = {
                            'HumanVal': False,
                            'x': 'nan',
                            'y': 'nan',
                            'ValDate': datetime.today().strftime('%Y-%m-%d'),
                            'Anomaly': False
                        }
                        new_row_df = pd.DataFrame([new_row_data], index=pd.MultiIndex.from_tuples([new_row_index], names=DB.index.names))
                        missing_files.append(new_row_df)
                    else:
                        print(f"Combination {combination} already exists in DB.")
                        

# Run the check
check_and_add_files(ROOT_PATH_IMAGES)

# Add missing files to DB if there are any
if missing_files:
    missing_df = pd.concat(missing_files)
    DB = pd.concat([DB, missing_df]).sort_index()
    
    # Save updated DB
    DB.to_pickle(os.path.join(f"{path}_new"))

pd.set_option('display.max.rows', None)

# print(DB)
print(DB.index.get_level_values("Campaign").unique())
