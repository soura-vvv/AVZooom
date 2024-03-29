import json

# Load the JSON data from the file
with open('zoroTest.json', 'r') as f:
    data = json.load(f)

# Iterate through each key-value pair in the JSON data
for key, value in data.items():
    # Copy the value of the "wav" key to create "noisy_wav"
    value['noisy_wav'] = value['wav']

# Write the updated JSON data back to the file
with open('zoroTest.json', 'w') as f:
    json.dump(data, f, indent=4)

