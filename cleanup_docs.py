"""
Documentation Cleanup & Archive Organization
==========================================
Consolidates old documentation files into organized archive structure
"""

import os
import shutil
from datetime import datetime

def organize_documentation():
    """Clean up and archive old documentation files"""
    
    print("📚 DOCUMENTATION CLEANUP & ORGANIZATION")
    print("=" * 50)
    
    # Ensure archive directory exists
    os.makedirs("docs/archive/old_readmes", exist_ok=True)
    os.makedirs("docs/archive/legacy", exist_ok=True)
    
    # Files to archive (if they exist)
    files_to_archive = [
        ("README_NEW.md", "docs/archive/old_readmes/"),
        ("README_backup.md", "docs/archive/old_readmes/"),
        ("README_STUNNING.md", "docs/archive/old_readmes/"),
    ]
    
    archived_count = 0
    
    for file_path, destination in files_to_archive:
        if os.path.exists(file_path):
            try:
                # Add timestamp to filename
                timestamp = datetime.now().strftime("%Y%m%d")
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                new_name = f"{base_name}_{timestamp}.md"
                dest_path = os.path.join(destination, new_name)
                
                shutil.move(file_path, dest_path)
                print(f"✅ Archived: {file_path} → {dest_path}")
                archived_count += 1
            except Exception as e:
                print(f"❌ Failed to archive {file_path}: {e}")
        else:
            print(f"⏭️ Skipped: {file_path} (not found)")
    
    print(f"\n📊 SUMMARY:")
    print(f"  📁 Files archived: {archived_count}")
    print(f"  🗂️ Archive location: docs/archive/")
    print(f"  📝 Main README: Clean and consolidated")
    
    print(f"\n✨ BENEFITS:")
    print(f"  🧹 Cleaner root directory")
    print(f"  📚 Organized documentation history")
    print(f"  🔍 Better repository navigation")
    print(f"  💼 Professional presentation")

if __name__ == "__main__":
    organize_documentation()
