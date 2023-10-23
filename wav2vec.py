import os
import pandas as pd
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import logging
import torch.nn.functional as F



logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def identify_speakers_and_generate_csv(directory_path, output_csv_path, similarity_threshold=0.65):
    try:
        logging.info("Loading SpeakerRecognition model...")
        # Load the pre-trained SpeakerRecognition model
        model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir_emozr")
        logging.info("SpeakerRecognition model loaded successfully.")

        # Initialize an empty list to store (file_name, speaker_id) pairs
        file_speaker_pairs = []
        speaker_id = 0
        csv_data = []
        

        logging.info("Processing WAV files and extracting embeddings...")
        # Process WAV files and extract embeddings
        for filename in os.listdir(directory_path):
            if filename.endswith(".wav"):
                logging.info(f"Processing file: {filename}")
             
                waveform = model.load_audio(os.path.join(directory_path, filename))
                embedding_np = waveform.unsqueeze(0)

                matched_speaker_id = None

                score = []
                speaker = []

                for _, _, existing_embedding, existing_speaker_id in file_speaker_pairs:
                   
                    similarity_score , pred = model.verify_batch(embedding_np,existing_embedding)
                    similarity_score = similarity_score[0][0]
                    pred = pred[0][0]
                    logging.info(f"Speaker ID {existing_speaker_id} - Similarity Score: {similarity_score} - Prediction: {pred}")
                    score.append(similarity_score)
                    speaker.append(existing_speaker_id)

                if len(file_speaker_pairs) !=0:

                    max_Score = max(score)
                    existing_speaker_id = speaker[score.index(max_Score)]
                        
                    if max_Score >= similarity_threshold:
                        matched_speaker_id = existing_speaker_id
                    

                logging.info(f"Length of File Speaker Pairs {len(file_speaker_pairs)}")

                # Assign a new speaker ID if no match is found
                if matched_speaker_id is None:
                    new_speaker_id = speaker_id + 1
                    speaker_id +=1
                    file_speaker_pairs.append((filename, new_speaker_id, embedding_np, new_speaker_id))
                    csv_data.append((filename, new_speaker_id))
                    logging.info(f"New speaker ID assigned: {new_speaker_id}")
                else:
                    file_speaker_pairs.append((filename, matched_speaker_id, embedding_np, matched_speaker_id))
                    csv_data.append((filename, matched_speaker_id))
                    logging.info(f"Matched with existing speaker ID: {matched_speaker_id}")

        logging.info("Creating CSV file...")
        # Create a DataFrame from the list of (file_name, speaker_id) pairs
        df = pd.DataFrame(csv_data, columns=["path", "spdx"])

        # Write the DataFrame to the CSV file using pandas
        df.to_csv(output_csv_path, index=False)

        logging.info(f"CSV file '{output_csv_path}' generated successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}",exc_info=True)

directory_path = "cv-corpus-5.1-2020-06-22/rw/updated_wavs"
output_csv_path = "cv-corpus-5.1-2020-06-22/rw/output.csv"
identify_speakers_and_generate_csv(directory_path, output_csv_path)
