import os
import pandas as pd

# Extract campaign names from the database
def get_campaigns_from_db(db_path):
    try:
        db = pd.read_pickle(db_path)
        campaigns_in_db = db.index.get_level_values('Campaign').unique().tolist()
        return campaigns_in_db
    except Exception as e:
        print(f"Error reading the database: {e}")
        return []

# Extract campaign names from directory
def get_campaigns_from_directory(dir_path):
    campaigns_in_directory = [
        name for name in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, name))
    ]
    return campaigns_in_directory

# Compare campaigns
def compare_campaigns(db_campaigns, dir_campaigns):
    missing_in_dir = [c for c in db_campaigns if c not in dir_campaigns]
    missing_in_db = [c for c in dir_campaigns if c not in db_campaigns]
    return missing_in_dir, missing_in_db

# Paths
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_DB")
dir_path = r"E:\CERN\BackupJuly2024\outputs"

# Get campaigns
db_campaigns = get_campaigns_from_db(db_path)
dir_campaigns = get_campaigns_from_directory(dir_path)

# Compare campaigns
missing_in_dir, missing_in_db = compare_campaigns(db_campaigns, dir_campaigns)

# Print results
print("Campaigns missing in directory:", missing_in_dir)
print("Campaigns missing in database:", missing_in_db)

