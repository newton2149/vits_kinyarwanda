import os
import shutil
import subprocess
import multiprocessing

def normalize_and_convert_audio(input_file):
    # Construct the output file path in the backup directory with a .wav extension
    output_file = os.path.join(backup_directory, os.path.splitext(os.path.basename(input_file))[0] + ".wav")

    # Check if the file exists in the backup directory
    if os.path.exists(output_file):
        print(f"Backup file for {input_file} already exists, skipping.")
    else:
        # Normalize audio to -27dBFS and resample to 22050 Hz and convert to WAV
        subprocess.run(['ffmpeg-normalize', input_file, '-o', output_file, '-c:a', 'pcm_s16le', '-ar', '22050', '-f'])
        print(f"Normalized and converted {input_file} to WAV: {output_file}")

if __name__ == "__main__":
    mp3_directory = "cv-corpus-5.1-2020-06-22/rw/selected_clips"
    root_directory = "cv-corpus-5.1-2020-06-22/rw"
    backup_directory = os.path.join(root_directory, "backup")
    file_extension = ".mp3"  # Change the file extension to '.mp3'

    # Create the backup directory if it doesn't exist
    os.makedirs(backup_directory, exist_ok=True)

    # Get a list of MP3 files
    mp3_files = [os.path.join(mp3_directory, file) for file in os.listdir(mp3_directory) if file.endswith(file_extension)]

    # Create a multiprocessing pool to parallelize the audio processing
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Use the multiprocessing pool to normalize and convert audio files concurrently
    pool.map(normalize_and_convert_audio, mp3_files)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
