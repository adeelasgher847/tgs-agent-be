"""
Fix Alembic revision issue by directly clearing alembic_version table
This bypasses Alembic's migration chain reading
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from app.core.config import settings


def fix_alembic_version():
    """Directly clear alembic_version table to fix broken revision references"""
    
    print("Fixing Alembic version table...")
    print(f"Database URL: {settings.DATABASE_URL[:60]}...")
    
    try:
        engine = create_engine(settings.DATABASE_URL)
        
        with engine.connect() as conn:
            # Check if alembic_version table exists
            check_table = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'alembic_version'
                );
            """))
            table_exists = check_table.fetchone()[0]
            
            if not table_exists:
                print("WARNING: alembic_version table does not exist. Creating it...")
                conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
                conn.commit()
                print("SUCCESS: Created alembic_version table")
            else:
                # Check current version
                result = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
                current_version = result.fetchone()
                
                if current_version:
                    print(f"Current broken version in DB: {current_version[0]}")
                else:
                    print("No version found in DB")
                
                # Clear the broken revision
                print("\nClearing broken revision from alembic_version table...")
                conn.execute(text("DELETE FROM alembic_version"))
                conn.commit()
                print("SUCCESS: Cleared alembic_version table!")
            
            # Verify it's empty
            result = conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
            count = result.fetchone()[0]
            print(f"Verified: alembic_version table now has {count} entries")
            
        engine.dispose()
        
        print("\n" + "="*60)
        print("SUCCESS: Database fixed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Error fixing alembic version: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    fix_alembic_version()
