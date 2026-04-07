import pandas as pd
import os

# Load the database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB_Lab_PC", "lab_PC_DB")
DB = pd.read_pickle(DB_PATH)

# Function to convert non-'nan' values to integers with -80 correction
def convert_to_int(value):
    if value == 'nan':
        return value  # Keep 'nan' as is
    else:
        value = float(value)
        if value % 160 != 0:
            value -= 80
        return int(value)  # Convert to float first, then to int

# Apply the conversion to the 'x' and 'y' columns
DB['x'] = DB['x'].apply(convert_to_int)
DB['y'] = DB['y'].apply(convert_to_int)

# Function to correct x and y values in bID
def correct_bID(bID):
    if bID == 'nan':
        return bID  # Keep 'nan' as is
    if isinstance(bID, str):
        parts = bID.split('-')
        if len(parts) == 2:
            x, y = parts
            x = convert_to_int(x)
            y = convert_to_int(y)
            return f"{x}-{y}"
    return bID  # Return the original bID if it's not in the expected format

# Function to update bID based on corrected x and y values
def update_bID(row):
    if row['x'] != 'nan' and row['y'] != 'nan':
        return f"{row['x']}-{row['y']}"
    return row['bID']  # Keep the original bID if x or y is 'nan'

# Assuming the index is a MultiIndex with levels Campaign, DUT, FileName, bID
if isinstance(DB.index, pd.MultiIndex):
    # Reset the index to work with columns
    DB_reset = DB.reset_index()
    
    # Correct the bID column
    DB_reset['bID'] = DB_reset['bID'].apply(correct_bID)
    
    # Update the bID column based on corrected x and y values
    DB_reset['bID'] = DB_reset.apply(update_bID, axis=1)
    
    # Set the index back to the original MultiIndex
    DB = DB_reset.set_index(['Campaign', 'DUT', 'FileName', 'bID'])

# Save the updated database back to the file (if needed)
DB.to_pickle(f"{DB_PATH}_new")

# Print the updated database
pd.set_option('display.max.rows', None)
print(DB)