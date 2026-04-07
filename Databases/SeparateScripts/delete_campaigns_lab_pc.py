import os
import pandas as pd

# Define file paths
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB_Lab_PC", "lab_PC_DB")
ROOT_PATH_IMAGES = r"E:\CERN\BackupJuly2024\Outputs_Lab_PC"

# Load the database
DB = pd.read_pickle(DB_PATH)

# Get existing campaigns in the database
existing_campaigns = set(DB.index.get_level_values("Campaign"))
print(existing_campaigns)
# Get campaigns from the disk folder
disk_campaigns = set(os.listdir(ROOT_PATH_IMAGES))

# Loop through each existing campaign and remove those not on disk
for existing_campaign in existing_campaigns:
    if existing_campaign not in disk_campaigns:
        print(f"{existing_campaign} removed from database.")
        # Create condition for campaigns that do not exist on disk
        condition = (DB.index.get_level_values("Campaign") == existing_campaign)
        # Filter out these campaigns from the database
        DB = DB[~condition]
    else:
        print(f"{existing_campaign} exists on disk.")
        
# Save updated DB
# DB.to_pickle(f"{DB_PATH}_new")   

