"""
Check migration status and verify tables
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from app.core.config import settings


def check_migration_status():
    """Check if migration was applied"""
    
    print("Checking migration status...")
    
    try:
        engine = create_engine(settings.DATABASE_URL)
        
        with engine.connect() as conn:
            # Check carrier table
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'carrier'
                )
            """))
            carrier_exists = result.fetchone()[0]
            print(f"Carrier table exists: {carrier_exists}")
            
            # Check alembic version
            result = conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
            version_row = result.fetchone()
            if version_row:
                print(f"Current alembic version: {version_row[0]}")
            else:
                print("No alembic version set (still at base)")
            
            # Check phonenumber columns
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'phonenumber' 
                AND column_name IN ('dialer_type', 'carrier_id', 'vicidial_campaign_id')
            """))
            columns = [row[0] for row in result.fetchall()]
            print(f"PhoneNumber Vicidial columns: {columns}")
            
            # Check callsession columns
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'callsession' 
                AND column_name IN ('dialer_type', 'vicidial_call_id', 'vicidial_lead_id')
            """))
            columns = [row[0] for row in result.fetchall()]
            print(f"CallSession Vicidial columns: {columns}")
            
        engine.dispose()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_migration_status()
