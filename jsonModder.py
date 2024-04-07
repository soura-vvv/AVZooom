import os
import json
import soundfile as sf

# Specify the directory containing the WAV files
folder_path = '9.90dBUnzoomed'

# Initialize an empty dictionary to store the JSON data
json_data = {}

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is a WAV file
    if file_name.endswith('.wav'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Get the length of the audio file
        audio_data, sample_rate = sf.read(file_path)
        audio_length = len(audio_data) / sample_rate
        
        # Construct the WAV and noisy WAV paths
        wav_path = f"{{data_root}}/{file_name}"
        noisy_wav_path = f"{{data_root}}/{file_name}"
        
        # Add the entry to the JSON data
        json_data[file_name] = {
            "wav": wav_path,
            "length": audio_length,
            "coordinates": [0.5, 0.5],
            "noisy_wav": noisy_wav_path
        }

# Write the JSON data to a file
with open("zorozoroInfer.json", "w") as json_file:
    json.dump(json_data, json_file, indent=4)





'''import json
import re
# Load the original JSON data
with open("zorozoroTrain.json", "r") as file:
    original_data = json.load(file)

# Iterate over each key-value pair
for key, value in original_data.items():
    # Extract the three numbers after "{data_folder}/"
    print(value["noisy_wav"])
    numbers_string = value["noisy_wav"].split("{data_root}/")[1].split("/Unzoomed/")[0]

    # Split the string by "_" and convert each part to a float
    
    numbers = [float(num) for num in numbers_string.split("_")]

    #print(numbers[1]/2)
    #exit()
    b = numbers[1]/2
    print("b")
    print(b)
    a = numbers[0]
    print(a)
    print("a")
    b = int(b) if b.is_integer() else b
    a = int(a) if a.is_integer() else a
    print("b")
    print(b)
    print("a")
    print(a)
    # Split the string by "Unzoomed/" to get the part after it
    parts = value["noisy_wav"].split("Unzoomed/")
    #print(parts)
    # Split the part after "Unzoomed/" by "_" to get the floats and the rest of the string
    old_a, rest = parts[1].split("_", 1)
    old_b,rest =rest.split("_", 1)
    #print(old_a)
    #print(old_b)
    #print(rest)
    print("PARTSS")
    print(parts[0])
    new_part = f"{a}_{b}_" + rest
    modified_noisy_wav = parts[0] + "Unzoomed/" + new_part
    original_data[key]["noisy_wav"] = modified_noisy_wav
    #print(original_data[key])
    

    #numbers = value["noisy_wav"].split("{data_root}/")[1].split("Unzoomed/")[1].split("_")[0:3]
    #numbers[2] = numbers[2][:len(numbers[2])-4]
    #print(numbers)
    #a, b, c = map(float, numbers)
    #print(a)
    #print(b)
    #print(c)
    # Modify the corresponding three numbers from the "Unzoomed" path
    #unzoomed_numbers = value["noisy_wav"].split("Unzoomed/")[1].split("_")[0:3]
    #print(unzoomed_numbers)
    #new_b, new_c = map(float, unzoomed_numbers[1:3])
    #new_b /= 2
    #new_c = 1.7

    # Construct the new "noisy_wav" value
    #new_noisy_wav = value["noisy_wav"].replace(f"{b}_{c}", f"{new_b}_{new_c}")

    # Update the value for "noisy_wav"
    #original_data[key]["noisy_wav"] = new_noisy_wav

# Write the modified data back to the JSON file
with open("zorozoroTrain.json", "w") as file:
    json.dump(original_data, file, indent=4)

#print(json.dumps(original_data, indent=4))'''
