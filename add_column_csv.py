import os
import pandas as pd

# CONSTANTS
data_dir = '.\\data'
output_dir = 'training_output'  # directory where the classifier(s) are stored
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

## Loop over datafiles
for filename in os.listdir(data_dir):
    if filename.endswith(".csv") and filename.startswith("walking-data"):
        filename_components = filename.split("-")  # split by the '-' character
        speaker = filename_components[2]
        identifier = filename_components[3].split(".")[0]
        
        csv_input = pd.read_csv(os.path.join(data_dir, filename))
        if speaker == "daniel" or speaker == "timothy":
            csv_input['gender'] = "male"
        else:
            csv_input['gender'] = "female"
        csv_input.to_csv('walking-data-{0}-{1}.csv'.format(speaker, identifier), index=False)