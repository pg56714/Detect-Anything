import shutil

shutil.copy("source_file.txt", "destination_file.txt")

shutil.copytree("source_directory", "destination_directory")

shutil.move("source_file.txt", "new_location/new_file.txt")

shutil.rmtree("directory_to_delete")

shutil.make_archive("backup", "zip", "directory_to_backup")

shutil.unpack_archive("backup.zip", "extracted_files")

try:
    shutil.copy("source_file.txt", "destination_file.txt")
except FileNotFoundError as e:
    print(f"Error: {e}")
except PermissionError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
