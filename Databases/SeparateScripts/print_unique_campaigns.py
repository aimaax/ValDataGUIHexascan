import os
import pandas as pd

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "updated_validation_DB")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DB_Lab_PC", "lab_PC_DB")

DB = pd.read_pickle(path)

pd.set_option('display.max.rows', None)

print(DB.index.get_level_values("Campaign").unique())

# ssh -X hgsensor@pclcd15.cern.ch

# /afs/cern.ch/user/h/hgsensor/ivcv_analysisworkflows_production/database/csv

#  rsync -aP maxdavid@lxplus.cern.ch:/eos/cms/store/group/hgcal/silicon/OpticalInspection/Outputs_Lab_PC/* Outputs_Lab_PC/