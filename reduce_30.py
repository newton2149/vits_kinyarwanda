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

# Create a directory for selected files in the destination folder if not exists
selected_directory = os.path.join(destination_directory, 'selected_clips_30')
os.makedirs(selected_directory, exist_ok=True)

# Move first 30 files from the source directory to the selected directory
selected_files = []
for idx, file in enumerate(os.listdir(source_directory)):
    if idx >= 30:
        break
    
    if file.endswith('.mp3'):
        file_path = os.path.join(source_directory, file)
        # Move the entire file to the selected directory
        destination_path = os.path.join(selected_directory, file)
        shutil.copyfile(file_path, destination_path)
        
        # Update the train.tsv file
        df = pd.read_csv(train_tsv_path, sep='\t')
        row = df[df['path']==file]
        

        # row_to_copy = row.iloc[0].tolist()
        # selected_files.append(row_to_copy)

        logging.info(f"Selected {idx + 1} files")

# Write the selected rows to the updated train.tsv file
selected_df = pd.DataFrame(selected_files)
selected_df.to_csv(os.path.join(destination_directory, 'train_updated_30.tsv'), sep='\t', index=False)

logging.info(f"Selected 30 files and updated train.tsv file is saved in {destination_directory}")
