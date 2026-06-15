import os
import re
import argparse
from pathlib import Path

def rename_videos(directory):
    """
    Rename videos from CineCameraActorN format to camNN format.
    
    Args:
        directory (str): Path to directory containing videos
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist.")
        return
    
    # Pattern to match CineCameraActorN files
    pattern = re.compile(r'^CineCameraActor(\d+)(.*)$')
    
    # Get all files matching the pattern
    files_to_rename = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                camera_num = int(match.group(1))
                extension = match.group(2)  # includes file extension and any suffix
                files_to_rename.append((file_path, camera_num, extension))
    
    if not files_to_rename:
        print(f"No files matching 'CineCameraActor*' pattern found in {directory}")
        return
    
    # Sort by camera number to ensure consistent renaming
    files_to_rename.sort(key=lambda x: x[1])
    
    print(f"Found {len(files_to_rename)} files to rename:")
    
    # Rename files
    for file_path, camera_num, extension in files_to_rename:
        # Create new filename with zero-padded camera number
        new_name = f"cam{camera_num:02d}{extension}"
        new_path = file_path.parent / new_name
        
        try:
            file_path.rename(new_path)
            print(f"Renamed: {file_path.name} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {file_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Rename video files from CineCameraActorN to camNN format")
    parser.add_argument("directory", help="Directory containing the video files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without actually renaming")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # Dry run mode - show what would be renamed
        directory = Path(args.directory)
        pattern = re.compile(r'^CineCameraActor(\d+)(.*)$')
        
        files_to_rename = []
        for file_path in directory.iterdir():
            if file_path.is_file():
                match = pattern.match(file_path.name)
                if match:
                    camera_num = int(match.group(1))
                    extension = match.group(2)
                    files_to_rename.append((file_path, camera_num, extension))
        
        files_to_rename.sort(key=lambda x: x[1])
        
        print(f"DRY RUN - Would rename {len(files_to_rename)} files:")
        for file_path, camera_num, extension in files_to_rename:
            new_name = f"cam{camera_num:02d}{extension}"
            print(f"  {file_path.name} -> {new_name}")
    else:
        rename_videos(args.directory)

if __name__ == "__main__":
    main()