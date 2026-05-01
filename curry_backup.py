#!/usr/bin/env python3
import os
import argparse
import glob
from datetime import datetime
from curry_core import Curry

def rotate_backups(backup_dir: str, keep: int):
    """Keep only the latest `keep` backups and delete the rest."""
    backups = sorted(glob.glob(os.path.join(backup_dir, "*.db")), reverse=True)
    if len(backups) > keep:
        for old_backup in backups[keep:]:
            try:
                os.remove(old_backup)
                print(f"Removed old backup: {old_backup}")
            except Exception as e:
                print(f"Failed to remove {old_backup}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Curry Database Backup Script")
    parser.add_argument("db_path", help="Path to the source Curry database")
    parser.add_argument("backup_dir", help="Directory to save the backup")
    parser.add_argument("--keep", type=int, default=5, help="Number of backups to keep (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.backup_dir):
        os.makedirs(args.backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_name = os.path.basename(args.db_path)
    if not db_name.endswith('.db'):
        db_name += '.db'

    basename = os.path.splitext(db_name)[0]
    if basename == ":memory:":
        basename = "memory"

    backup_file = os.path.join(args.backup_dir, f"{basename}_backup_{timestamp}.db")

    print(f"Starting backup of {args.db_path} to {backup_file}...")
    try:
        curry = Curry(args.db_path)
        curry.backup(backup_file)
        curry.close()
        print("Backup completed successfully.")
    except Exception as e:
        print(f"Backup failed: {e}")
        return

    rotate_backups(args.backup_dir, args.keep)

if __name__ == "__main__":
    main()
