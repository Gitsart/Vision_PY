import os

def rename_files(directory, prefix):
    for count, filename in enumerate(os.listdir(directory), start=1):
        ext = os.path.splitext(filename)[1]  # Keep the file extension
        new_name = f"{prefix}_{count}{ext}"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
        print(f"Renamed {filename} to {new_name}")

# Paths to the good and bad folders
good_folder = r"C:\Users\TECQNIO\project_folder\dataset\good"
bad_folder = r"C:\Users\TECQNIO\project_folder\dataset\bad"

# Rename files in each folder
rename_files(good_folder, "good_image")
rename_files(bad_folder, "bad_image")
