# Database Backups

This directory contains database backup files created by the `scripts/db_backup.py` utility.

## Usage

```bash
# Create a backup
python scripts/db_backup.py backup

# Create a backup with custom filename
python scripts/db_backup.py backup backups/my_custom_backup.sql

# List available backups
python scripts/db_backup.py list

# Restore from backup
python scripts/db_backup.py restore backups/msrk_backup_20241208_143022.sql
```

## Backup Files

Backup files are named with the pattern: `msrk_backup_YYYYMMDD_HHMMSS.sql`

**Important:** These are full database dumps. Restoring will replace all existing data.

## Git Ignore

Backup files are excluded from git via `.gitignore` to prevent accidentally committing sensitive data.
