import pandas as pd
from datetime import datetime

# Example DataFrame with a multi-index
data = {
    'Campaign': ['Preseries_December2022', 'Preseries_December2022'],
    'DUT': [100163, 100163],
    'FileName': ['annotated_image_pad6_step8', 'annotated_image_pad6_step8'],
    'bID': ['2240-1760', '2400-1760'],
    'HumanVal': [False, False],
    'x': [2240, 2400],
    'y': [1760, 1760],
    'ValDate': ['2024-08-10', '2024-08-10'],
    'Anomaly': [True, True]
}

# Convert to DataFrame and set multi-index
df = pd.DataFrame(data)
df.set_index(['Campaign', 'DUT', 'FileName', 'bID'], inplace=True)

# New row data
new_row_index = ('Preseries_December2022', 100163, 'annotated_image_pad6_step8', '0-0')
new_row_data = {
    'HumanVal': False,
    'x': 0,
    'y': 0,
    'ValDate': datetime.today().strftime('%Y-%m-%d'),
    'Anomaly': True
}

# Append the new row
new_row_df = pd.DataFrame([new_row_data], index=pd.MultiIndex.from_tuples([new_row_index], names=df.index.names))
df = pd.concat([df, new_row_df])

print(df)
