import os
import pandas as pd

def check_files_in_db():
    # db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "updated_validation_DB")
    # base_dir = r"E:\CERN\BackupJuly2024\outputs"
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB_Lab_PC", "lab_PC_DB")
    base_dir = r"E:\CERN\BackupJuly2024\Outputs_Lab_PC"
    df = pd.read_pickle(db_path)  # Adjust if your DB is in a different format

    missing_files = []
    # Assumes the DataFrame has a multi-index with levels ["Campaign", "DUT", "FileName"]
    for (campaign, dut, filename, _) in df.index.unique():
        file_path = os.path.join(base_dir, campaign, dut, f"{filename}.npy")
        if not os.path.isfile(file_path):
            print("file deleted in database: ", file_path)
            missing_files.append(file_path)
            # Create a condition to match the first three levels
            condition = (
                (df.index.get_level_values("Campaign") == campaign) &
                (df.index.get_level_values("DUT") == dut) &
                (df.index.get_level_values("FileName") == filename)
            )
            
            # self.database = self.database.drop((campaign, dut, filename)).sort_index()
            df = df[~condition].sort_index()
            print(missing_files)

    print("Missing files:")
    for mf in missing_files:
        print(mf)
    df.to_pickle(f"{db_path}_new")

if __name__ == "__main__":
    check_files_in_db()
