#!/usr/bin/env python3
"""
Directory cleanup script to archive auxiliary files in environment directories.
Keeps only core environment files in the root, moves everything else to archive/
"""

import os
import shutil
import sys
from pathlib import Path

# Core files that should remain in the environment root directory
CORE_FILES = {
    'config.yaml',
    'env_desc.txt',
    'env_implement.txt',
    'env_main.py',
    'env_obs.py',
    'env_generate.py',
    'env_main_use.py',
    'agent_instruction.txt',
    'action_space.txt',
    'env_validator.py',
    'level_max_rewards.json',
    'task_completion_summary.json',
    'cost.csv',
}

# Core directories that should remain in the environment root directory
CORE_DIRS = {
    'levels',
    'archive'  # Don't move archive directory itself
}

def archive_auxiliary_files(env_dir: str, dry_run: bool = False):
    """
    Archive auxiliary files in an environment directory.
    
    Args:
        env_dir: Path to environment directory
        dry_run: If True, only print what would be moved without actually moving
    """
    env_path = Path(env_dir).resolve()
    if not env_path.exists() or not env_path.is_dir():
        print(f"Error: Directory {env_path} does not exist")
        return False
    
    print(f"Processing environment directory: {env_path}")
    
    # Create archive directory if it doesn't exist
    archive_path = env_path / 'archive'
    if not dry_run:
        archive_path.mkdir(exist_ok=True)
    else:
        print(f"Would create: {archive_path}")
    
    # Get all items in the environment directory
    all_items = list(env_path.iterdir())
    files_to_move = []
    
    for item in all_items:
        if item.name in CORE_FILES or item.name in CORE_DIRS:
            continue
            
        # Skip hidden files/directories (starting with .)
        if item.name.startswith('.'):
            continue
            
        files_to_move.append(item)
    
    if not files_to_move:
        print("No auxiliary files found to archive")
        return True
    
    print(f"\nFiles/directories to move to archive:")
    for item in files_to_move:
        item_type = "DIR" if item.is_dir() else "FILE"
        print(f"  {item_type}: {item.name}")
    
    if dry_run:
        print(f"\nDRY RUN - No files were actually moved")
        return True
    
    # Move files to archive
    moved_count = 0
    for item in files_to_move:
        try:
            dest_path = archive_path / item.name
            if dest_path.exists():
                # If destination exists, rename with timestamp
                import time
                timestamp = int(time.time())
                dest_path = archive_path / f"{item.name}.{timestamp}"
            
            shutil.move(str(item), str(dest_path))
            moved_count += 1
            print(f"Moved: {item.name} -> archive/{dest_path.name}")
        except Exception as e:
            print(f"Error moving {item.name}: {e}")
    
    print(f"\nArchived {moved_count} files/directories")
    
    # Show final directory structure
    print(f"\nFinal directory structure:")
    for item in sorted(env_path.iterdir()):
        if item.is_dir():
            print(f"  DIR:  {item.name}/")
        else:
            print(f"  FILE: {item.name}")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Archive auxiliary files in environment directories')
    parser.add_argument('env_dir', help='Path to environment directory')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    success = archive_auxiliary_files(args.env_dir, args.dry_run)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
