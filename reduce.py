import os
import shutil
import pandas as pd
from pydub import AudioSegment
import logging

# Source and destination directories
source_directory = '/home/navneeth/EgoPro/deep_learning/vits_new/cv-corpus-5.1-2020-06-22/rw/clips'
destination_directory = '/home/navneeth/EgoPro/deep_learning/vits_new/cv-corpus-5.1-2020-06-22/rw/'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Path to train.tsv file
train_tsv_path = '/home/navneeth/EgoPro/deep_learning/vits_new/cv-corpus-5.1-2020-06-22/rw/train.tsv'

# Desired total duration in seconds (50 hours)
desired_duration = 240 # 50 hours in seconds

# Create a directory for selected files in the destination folder if not exists
selected_directory = os.path.join(destination_directory, 'selected_clips_30')
os.makedirs(selected_directory, exist_ok=True)


# Move files and update train.tsv until the total duration in the destination folder is less than or equal to 50 hours
current_duration = 0
selected_files = []

# Read the train.tsv file into a DataFrame
df = pd.read_csv(train_tsv_path, sep='\t')
df = df[df['gender'] == 'female']


for file in os.listdir(source_directory):
    if file.endswith('.mp3'):
        file_path = os.path.join(source_directory, file)
        audio_duration = AudioSegment.from_file(file_path).duration_seconds
        
        if current_duration + audio_duration <= desired_duration:
            # Move the entire file to the selected directory
            destination_path = os.path.join(selected_directory, file)
            shutil.copyfile(file_path, destination_path)
            current_duration += audio_duration
            
            # Update the train.tsv file
            row = df[df['path'].str.endswith(file)]
            if not row.empty:
                selected_files.append(row.iloc[0])

            logging.info(f"Current duration: {current_duration} seconds")
        else:
            break

# Write the selected rows to the updated train.tsv file
selected_df = pd.DataFrame(selected_files)
selected_df.to_csv(os.path.join(destination_directory, 'train_updated_30.tsv'), sep='\t', index=False)

logging.info(f"Total duration of selected files in the destination folder: {current_duration} seconds")
logging.info(f"Updated train.tsv file is saved in {destination_directory}")
