#!/usr/bin/env python3
"""
Database migration helper script for MSRK v3
Provides a unified interface for common Prisma database operations.
"""

import subprocess
import sys
from typing import Optional

def run_command(cmd: str, description: str) -> bool:
    """Run a command and handle errors gracefully."""
    print(f"📊 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def check_prisma():
    """Check if Prisma CLI is available."""
    try:
        subprocess.run(["npx", "prisma", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Prisma CLI not found. Please install it with: npm install -g prisma")
        return False

def generate_client():
    """Generate Prisma client."""
    return run_command("npx prisma generate", "Generating Prisma client")

def push_schema():
    """Push schema changes to database (development)."""
    return run_command("npx prisma db push", "Pushing schema to database")

def create_migration(name: Optional[str] = None) -> bool:
    """Create a new migration."""
    if not name:
        name = input("Enter migration name: ").strip()
    if not name:
        print("❌ Migration name required")
        return False
    
    cmd = f'npx prisma migrate dev --name "{name}"'
    return run_command(cmd, f"Creating migration: {name}")

def deploy_migrations():
    """Deploy pending migrations (production)."""
    return run_command("npx prisma migrate deploy", "Deploying migrations")

def reset_database():
    """Reset database (WARNING: destructive)."""
    confirm = input("⚠️  This will DELETE ALL DATA. Type 'RESET' to confirm: ").strip()
    if confirm != "RESET":
        print("❌ Database reset cancelled")
        return False
    
    return run_command("npx prisma migrate reset --force", "Resetting database")

def studio():
    """Open Prisma Studio."""
    print("🎨 Opening Prisma Studio...")
    subprocess.Popen(["npx", "prisma", "studio"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Prisma Studio opened in background")

def main():
    """Main CLI interface."""
    if not check_prisma():
        return 1
    
    if len(sys.argv) < 2:
        print("""
📊 MSRK v3 Database Migration Helper

Usage: python scripts/db_migrate.py <command>

Commands:
  generate     Generate Prisma client
  push         Push schema to database (development)
  migrate      Create new migration
  deploy       Deploy migrations (production)
  reset        Reset database (destructive)
  studio       Open Prisma Studio
        """)
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "generate":
        success = generate_client()
    elif command == "push":
        success = push_schema()
    elif command == "migrate":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        success = create_migration(name)
    elif command == "deploy":
        success = deploy_migrations()
    elif command == "reset":
        success = reset_database()
    elif command == "studio":
        studio()
        success = True
    else:
        print(f"❌ Unknown command: {command}")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
