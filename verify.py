# import os
# from scipy.io import wavfile

# # Specify the directory containing the WAV files
# directory_path = '/home/navneeth/EgoPro/deep_learning/vits_new/cv-corpus-5.1-2020-06-22/rw/backup'

# # Get a list of all files in the directory
# wav_files = [file for file in os.listdir(directory_path) if file.endswith('.wav')]

# # Iterate through the WAV files and print their sample rates
# for wav_file in wav_files:
#     wav_file_path = os.path.join(directory_path, wav_file)
#     try:
#         sample_rate, audio_data = wavfile.read(wav_file_path)
#         print(f'Sample rate of {wav_file}: {sample_rate} Hz')
#     except Exception as e:
#         print(f'Error reading {wav_file}: {str(e)}')


import os

def count_wav_files(directory_path):
    """
    Count the number of WAV files in the specified directory.

    Parameters:
        directory_path (str): Path to the directory containing WAV files.

    Returns:
        int: Number of WAV files in the directory.
    """
    # Initialize a counter for WAV files
    wav_file_count = 0
    
    # Iterate through the files in the directory and count WAV files
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            wav_file_count += 1
    
    return wav_file_count

def count_mp3_files(directory_path):
    """
    Count the number of WAV files in the specified directory.

    Parameters:
        directory_path (str): Path to the directory containing WAV files.

    Returns:
        int: Number of WAV files in the directory.
    """
    # Initialize a counter for WAV files
    wav_file_count = 0
    
    # Iterate through the files in the directory and count WAV files
    for filename in os.listdir(directory_path):
        if filename.endswith(".mp3"):
            wav_file_count += 1
    
    return wav_file_count

# Example usage
directory_path = "/home/navneeth/EgoPro/deep_learning/vits_new/cv-corpus-5.1-2020-06-22/rw/backup"
number_of_wav_files = count_wav_files(directory_path)
print(f"Number of WAV files in the directory: {number_of_wav_files}")

directory_path = "/home/navneeth/EgoPro/deep_learning/vits_new/cv-corpus-5.1-2020-06-22/rw/clips"
number_of_wav_files = count_mp3_files(directory_path)
print(f"Number of MP3 files in the directory: {number_of_wav_files}")
