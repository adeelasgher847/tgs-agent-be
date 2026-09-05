"""
Create or update a SysAdmin user.

Usage:
    source venv/bin/activate
    python -m app.scripts.create_sysadmin \
        --email admin@tgs.ai \
        --role SUPER_ADMIN

The password is prompted interactively (not passed as a CLI arg).
"""
from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import select


def main():
    parser = argparse.ArgumentParser(description="Create a SysAdmin Portal user")
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", choices=["ADMIN", "SUPER_ADMIN"], default="SUPER_ADMIN")
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    if not password:
        print("Error: password cannot be empty", file=sys.stderr)
        sys.exit(1)

    from app.db.session import SessionLocal
    from app.models.sysadmin_user import SysAdminUser
    from app.sysadmin.security import hash_password

    db = SessionLocal()
    try:
        existing = db.execute(select(SysAdminUser).where(SysAdminUser.email == args.email)).scalar_one_or_none()
        if existing:
            existing.hashed_password = hash_password(password)
            existing.role = args.role
            existing.is_active = True
            db.commit()
            print(f"Updated sysadmin: {args.email} ({args.role})")
        else:
            user = SysAdminUser(
                email=args.email,
                hashed_password=hash_password(password),
                role=args.role,
            )
            db.add(user)
            db.commit()
            print(f"Created sysadmin: {args.email} ({args.role})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
