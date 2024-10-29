import os

# Path to the main directory containing subdirectories
main_dir = "./data/Stanford Dogs Dataset"


def rename_folder(main_dir):
    # Iterate through all subdirectories in the main directory
    for subdir in os.listdir(main_dir):
        # Construct full path to the subdirectory
        subdir_path = os.path.join(main_dir, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Split the directory name by '-'
            parts = subdir.split("-")

            # Ensure there are at least 2 parts after split
            if len(parts) > 1:
                # Get the second part of the split
                new_name = parts[1]

                # Construct the new full path
                new_subdir_path = os.path.join(main_dir, new_name.upper())

                # Rename the subdirectory
                os.rename(subdir_path, new_subdir_path)
                print(f"Renamed: {subdir} -> {new_name}")


def remove_zone_identifier_files(base_dir):
    """
    Recursively remove all files containing "Zone.Identifier" in their names.

    Args:
        base_dir (str): The path to the base directory to start the search.
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "Zone.Identifier" in file:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Could not remove {file_path}: {e}")


def rename_files_sequentially(base_dir):
    """
    Rename files within each subdirectory of the base_dir sequentially.

    Args:
        base_dir (str): The path to the base directory containing subdirectories.
    """
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            files.sort()  # Ensure consistent ordering

            for i, filename in enumerate(files):
                file_path = os.path.join(subdir_path, filename)

                # Ensure it's a file before renaming
                if os.path.isfile(file_path):
                    file_extension = os.path.splitext(filename)[1]
                    new_filename = f"{i + 1}{file_extension}"
                    new_file_path = os.path.join(subdir_path, new_filename)

                    try:
                        os.rename(file_path, new_file_path)
                        print(f"Renamed: {file_path} to {new_file_path}")
                    except Exception as e:
                        print(f"Could not rename {file_path} to {new_file_path}: {e}")


# # Rename folders
# rename_folder(main_dir)
#
# # Remove Zone.Identifier files
# remove_zone_identifier_files(main_dir)

# # Rename files sequentially
# rename_files_sequentially(main_dir)
