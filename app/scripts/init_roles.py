#!/usr/bin/env python3
"""
Script to initialize required roles in the database.
Run: python app/scripts/init_roles.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.role import Role

def create_required_roles():
    """Create all required roles for the application"""
    db: Session = SessionLocal()
    
    try:
        # Canonical RBAC roles (admin > manager > config_only > read_only,
        # billing_only outside the chain) — see docs/rbac-matrix.md.
        # 'owner' and 'member' were retired in migration 9f3a2c7e5d41: the
        # workspace creator is tracked via user_tenant_association.is_creator
        # instead of a role name.
        required_roles = [
            {
                "name": "admin",
                "description": "Administrator role with full access"
            },
            {
                "name": "manager",
                "description": "Full operational access; cannot manage members or billing"
            },
            {
                "name": "config_only",
                "description": "Configure workspace settings; cannot manage users"
            },
            {
                "name": "read_only",
                "description": "Read-only access; blocked from mutating endpoints"
            },
            {
                "name": "billing_only",
                "description": "Access limited to billing endpoints (usage, pricing)"
            },
        ]
        
        created_count = 0
        existing_count = 0
        
        for role_data in required_roles:
            # Check if role already exists
            existing_role = db.query(Role).filter(Role.name == role_data["name"]).first()
            
            if existing_role:
                print(f"✅ Role '{role_data['name']}' already exists")
                existing_count += 1
            else:
                # Create new role
                new_role = Role(
                    name=role_data["name"],
                    description=role_data["description"]
                )
                db.add(new_role)
                print(f"🆕 Created role '{role_data['name']}': {role_data['description']}")
                created_count += 1
        
        db.commit()
        
        print(f"\n📊 Summary:")
        print(f"   - Created: {created_count} roles")
        print(f"   - Already existed: {existing_count} roles")
        print(f"   - Total roles: {created_count + existing_count}")
        
        # List all roles
        all_roles = db.query(Role).all()
        print(f"\n📋 All roles in database:")
        for role in all_roles:
            print(f"   - {role.name}: {role.description}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Initializing required roles...")
    create_required_roles()
    print("\n✅ Role initialization complete!")
