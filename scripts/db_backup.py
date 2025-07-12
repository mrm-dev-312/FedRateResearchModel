#!/usr/bin/env python3
"""
Database backup and restore utilities for MSRK v3
Provides PostgreSQL backup/restore functionality for development workflows.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

def get_db_url() -> Optional[str]:
    """Get database URL from environment."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL environment variable not found")
        print("   Please ensure your .env file is properly configured")
        return None
    return db_url

def run_command(cmd: str, description: str) -> bool:
    """Run a command and handle errors gracefully."""
    print(f"💾 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            # Only show first few lines to avoid spam
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 5:
                print(f"   Output: {output_lines[0]}...({len(output_lines)} lines total)")
            else:
                print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def create_backup(output_file: Optional[str] = None) -> bool:
    """Create a database backup using pg_dump."""
    db_url = get_db_url()
    if not db_url:
        return False
    
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"backups/msrk_backup_{timestamp}.sql"
    
    # Ensure backups directory exists
    backup_dir = Path(output_file).parent
    backup_dir.mkdir(exist_ok=True)
    
    # Use pg_dump with connection string
    cmd = f'pg_dump "{db_url}" --no-password --verbose --file="{output_file}"'
    
    success = run_command(cmd, f"Creating backup: {output_file}")
    
    if success:
        file_size = Path(output_file).stat().st_size / 1024 / 1024  # MB
        print(f"📁 Backup saved: {output_file} ({file_size:.1f} MB)")
    
    return success

def restore_backup(backup_file: str) -> bool:
    """Restore a database backup using psql."""
    db_url = get_db_url()
    if not db_url:
        return False
    
    if not Path(backup_file).exists():
        print(f"❌ Backup file not found: {backup_file}")
        return False
    
    # Confirm destructive operation
    confirm = input(f"⚠️  This will REPLACE all data with backup from {backup_file}. Type 'RESTORE' to confirm: ").strip()
    if confirm != "RESTORE":
        print("❌ Restore cancelled")
        return False
    
    # Use psql to restore
    cmd = f'psql "{db_url}" --file="{backup_file}"'
    
    return run_command(cmd, f"Restoring backup: {backup_file}")

def list_backups() -> None:
    """List available backup files."""
    backup_dir = Path("backups")
    
    if not backup_dir.exists():
        print("📁 No backups directory found")
        return
    
    backup_files = list(backup_dir.glob("*.sql"))
    
    if not backup_files:
        print("📁 No backup files found in backups/")
        return
    
    print(f"📁 Found {len(backup_files)} backup files:")
    for backup_file in sorted(backup_files, reverse=True):  # newest first
        stat = backup_file.stat()
        size_mb = stat.st_size / 1024 / 1024
        modified = datetime.fromtimestamp(stat.st_mtime)
        print(f"   {backup_file.name} ({size_mb:.1f} MB, {modified.strftime('%Y-%m-%d %H:%M')})")

def main():
    """CLI interface for backup/restore operations."""
    if len(sys.argv) < 2:
        print("""
💾 MSRK v3 Database Backup/Restore

Usage: python scripts/db_backup.py <command> [options]

Commands:
  backup [filename]    Create database backup
  restore <filename>   Restore database from backup  
  list                 List available backups

Examples:
  python scripts/db_backup.py backup
  python scripts/db_backup.py backup backups/my_backup.sql
  python scripts/db_backup.py restore backups/msrk_backup_20241208_143022.sql
  python scripts/db_backup.py list
        """)
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "backup":
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        success = create_backup(output_file)
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ Backup filename required for restore")
            return 1
        backup_file = sys.argv[2]
        success = restore_backup(backup_file)
    elif command == "list":
        list_backups()
        success = True
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
