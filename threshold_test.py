import subprocess

# Define the range of threshold values you want to test
threshold_values = range(50, 201, 10)  # Example range from 50 to 200 in increments of 10

# Loop through each threshold value
for threshold in threshold_values:
    print(f"\nTesting with threshold: {threshold}")
    # Run `image_ready.py` with the current threshold
    subprocess.run(["python", "image_reader.py", str(threshold)])
