import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "updated_validation_DB_Lab_PC")
# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB_Lab_PC", "lab_PC_DB")
DB = pd.read_pickle(DB_PATH)

# DB = DB.drop(['OxideVariants_2023', 'TCT_October2022'], errors='ignore')
# DB = DB.rename(columns={'AI': 'HumanVal', 'Normal': 'Anomaly'})
# DB['HumanVal'] = DB['HumanVal'].replace({1: False, 0: True})
# DB['Anomaly'] = ~DB['Anomaly']
# DB['HumanVal'] = False
# DB = DB.sort_index()

# # Define the condition to filter rows
# campaign = 'Production_HD_200_April_2024'
# dut = '250066'
# filename = 'annotated_image_pad82_step73'

# # Filter the database using .loc with a partial index match
# filtered_db = DB.loc[(campaign, dut, filename)]

# # Display the filtered database
# print(filtered_db)

pd.set_option('display.max.rows', None)
print(DB)

# anom_val = DB[DB.x != 'nan']
# anom_val = DB[(DB.index.get_level_values('Campaign') == "Production_LD_300") & (DB.x != 'nan')]
# unique_filenames = anom_val.index.get_level_values('FileName').unique()[:10]
# print(unique_filenames)
# anom_val = anom_val.loc[anom_val.index.get_level_values('FileName').isin(unique_filenames)]

# normal_val = DB[DB.x == 'nan']

# print("normal val\n", normal_val)
# print("anom val\n", anom_val)

# db_path_store = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manually_checked_validation_DB")
# db_path_store = os.path.join(os.path.dirname(os.path.abspath(__file__)), "10FileNames_validation_DB")

# anom_val.to_pickle(db_path_store)
# DB.to_pickle(f"{DB_PATH}_new")