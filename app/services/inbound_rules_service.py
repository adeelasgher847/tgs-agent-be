"""Inbound Rules & Blocklist Rule Sets service."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.call_flow import CallFlow
from app.models.inbound_rule import InboundRule, InboundRuleSet
from app.schemas.inbound_rule import (
    InboundRuleImportRequest,
    InboundRuleImportResponse,
    InboundRuleItem,
    InboundRuleSetCreate,
    InboundRuleSetListItem,
    InboundRuleSetResponse,
    InboundRuleSetUpdate,
    normalize_phone_digits,
)


class InboundRulesService:
    def list_rule_sets(
        self, db: Session, tenant_id: uuid.UUID
    ) -> List[InboundRuleSetListItem]:
        """List all non-deleted rule sets with active rules count for tenant."""
        rule_sets = (
            db.execute(
                select(InboundRuleSet)
                .where(
                    InboundRuleSet.tenant_id == tenant_id,
                    InboundRuleSet.is_deleted.is_(False),
                )
                .order_by(InboundRuleSet.created_at.desc())
            )
            .scalars()
            .all()
        )

        result: List[InboundRuleSetListItem] = []
        for rs in rule_sets:
            count = (
                db.execute(
                    select(func.count(InboundRule.id)).where(
                        InboundRule.rule_set_id == rs.id,
                        InboundRule.tenant_id == tenant_id,
                        InboundRule.is_deleted.is_(False),
                    )
                ).scalar()
                or 0
            )
            result.append(
                InboundRuleSetListItem(
                    id=rs.id,
                    name=rs.name,
                    description=rs.description,
                    rules_count=count,
                    created_at=rs.created_at,
                    updated_at=rs.updated_at,
                )
            )
        return result

    def get_rule_set(
        self, db: Session, tenant_id: uuid.UUID, rule_set_id: uuid.UUID
    ) -> InboundRuleSetResponse:
        """Get single rule set with full rules list."""
        rs = self._get_rule_set_or_404(db, tenant_id, rule_set_id)
        return self._to_rule_set_response(db, rs)

    def create_rule_set(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        user_id: Optional[uuid.UUID],
        body: InboundRuleSetCreate,
    ) -> InboundRuleSetResponse:
        """Create new rule set and optionally its initial rules."""
        rs = InboundRuleSet(
            tenant_id=tenant_id,
            name=body.name,
            description=body.description,
            created_by=user_id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(rs)
        db.flush()

        seen_digits = set()
        for item in body.rules:
            digits = normalize_phone_digits(item.phone_number_pattern)
            if not digits or digits in seen_digits:
                continue
            seen_digits.add(digits)
            rule = InboundRule(
                tenant_id=tenant_id,
                rule_set_id=rs.id,
                phone_number_pattern=item.phone_number_pattern.strip(),
                normalized_digits=digits,
                label=item.label,
                action=item.action or "deny",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(rule)

        db.commit()
        db.refresh(rs)
        return self._to_rule_set_response(db, rs)

    def update_rule_set(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        rule_set_id: uuid.UUID,
        body: InboundRuleSetUpdate,
    ) -> InboundRuleSetResponse:
        """Update rule set name/description and batch replace rules if provided."""
        rs = self._get_rule_set_or_404(db, tenant_id, rule_set_id)

        if body.name is not None:
            rs.name = body.name
        if body.description is not None:
            rs.description = body.description
        rs.updated_at = datetime.now(timezone.utc)

        if body.rules is not None:
            # Mark existing rules as deleted or delete
            existing_rules = (
                db.execute(
                    select(InboundRule).where(
                        InboundRule.rule_set_id == rs.id,
                        InboundRule.tenant_id == tenant_id,
                    )
                )
                .scalars()
                .all()
            )
            for er in existing_rules:
                db.delete(er)
            db.flush()

            seen_digits = set()
            for item in body.rules:
                digits = normalize_phone_digits(item.phone_number_pattern)
                if not digits or digits in seen_digits:
                    continue
                seen_digits.add(digits)
                rule = InboundRule(
                    tenant_id=tenant_id,
                    rule_set_id=rs.id,
                    phone_number_pattern=item.phone_number_pattern.strip(),
                    normalized_digits=digits,
                    label=item.label,
                    action=item.action or "deny",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                db.add(rule)

        db.commit()
        db.refresh(rs)
        return self._to_rule_set_response(db, rs)

    def delete_rule_set(
        self, db: Session, tenant_id: uuid.UUID, rule_set_id: uuid.UUID
    ) -> None:
        """Soft delete a rule set and nullify flow references."""
        rs = self._get_rule_set_or_404(db, tenant_id, rule_set_id)
        rs.is_deleted = True
        rs.updated_at = datetime.now(timezone.utc)

        # Mark rules as deleted
        rules = (
            db.execute(
                select(InboundRule).where(
                    InboundRule.rule_set_id == rs.id,
                    InboundRule.tenant_id == tenant_id,
                )
            )
            .scalars()
            .all()
        )
        for r in rules:
            r.is_deleted = True
            r.updated_at = datetime.now(timezone.utc)

        # Detach from any call flows
        flows = (
            db.execute(
                select(CallFlow).where(
                    CallFlow.tenant_id == tenant_id,
                    CallFlow.inbound_rule_set_id == rs.id,
                )
            )
            .scalars()
            .all()
        )
        for f in flows:
            f.inbound_rule_set_id = None
            f.updated_at = datetime.now(timezone.utc)

        db.commit()

    def import_rules_from_text(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        user_id: Optional[uuid.UUID],
        body: InboundRuleImportRequest,
    ) -> InboundRuleImportResponse:
        """Bulk import rules from multiline text / CSV string."""
        raw_text = (body.raw_text or "").strip()
        if not raw_text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="raw_text cannot be empty",
            )

        # Resolve or create target rule set
        if body.rule_set_id is not None:
            rs = self._get_rule_set_or_404(db, tenant_id, body.rule_set_id)
        else:
            name = (
                body.new_rule_set_name
                or f"Imported Blocklist - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
            )
            rs = InboundRuleSet(
                tenant_id=tenant_id,
                name=name,
                description=body.new_rule_set_description,
                created_by=user_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(rs)
            db.flush()

        # Fetch existing normalized digits in this rule set
        existing_rules = (
            db.execute(
                select(InboundRule).where(
                    InboundRule.rule_set_id == rs.id,
                    InboundRule.tenant_id == tenant_id,
                    InboundRule.is_deleted.is_(False),
                )
            )
            .scalars()
            .all()
        )
        existing_digits = {r.normalized_digits for r in existing_rules}

        # Parse lines / CSV
        parsed_rows: List[Tuple[str, Optional[str]]] = []
        reader = csv.reader(io.StringIO(raw_text))
        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue
            phone_raw = row[0].strip()
            label_raw = row[1].strip() if len(row) > 1 else None
            # Skip header lines if present (e.g. 'phone_number,label', 'Phone,Label', 'caller_id,tag')
            clean_header = (
                phone_raw.lower()
                .replace(" ", "")
                .replace("_", "")
                .replace("-", "")
            )
            if clean_header in (
                "phone",
                "phonenumber",
                "phonenumbers",
                "number",
                "numbers",
                "callerid",
                "telephone",
                "telephonenumber",
            ):
                continue
            parsed_rows.append((phone_raw, label_raw if label_raw else None))

        imported_count = 0
        skipped_count = 0
        seen_import_digits = set()

        for phone_raw, label in parsed_rows:
            digits = normalize_phone_digits(phone_raw)
            if not digits:
                skipped_count += 1
                continue
            if digits in existing_digits or digits in seen_import_digits:
                skipped_count += 1
                continue

            seen_import_digits.add(digits)
            clean_label = label.strip()[:100] if label and label.strip() else None
            rule = InboundRule(
                tenant_id=tenant_id,
                rule_set_id=rs.id,
                phone_number_pattern=phone_raw[:50],
                normalized_digits=digits[:50],
                label=clean_label,
                action="deny",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(rule)
            imported_count += 1

        rs.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(rs)

        resp = self._to_rule_set_response(db, rs)
        return InboundRuleImportResponse(
            rule_set=resp,
            imported_count=imported_count,
            skipped_count=skipped_count,
            total_rules_count=resp.rules_count,
        )

    def is_number_blocked(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        rule_set_id: uuid.UUID,
        phone_number: str,
    ) -> Tuple[bool, Optional[InboundRule]]:
        """
        Check if a caller phone number is denied in the given rule set.
        Matching handles exact digits, 10-digit national variants (with/without country code 1).
        """
        if not phone_number or not rule_set_id:
            return False, None

        digits = normalize_phone_digits(phone_number)
        if not digits:
            return False, None

        # Build search variants (e.g. 15551234567, 5551234567)
        candidates = {digits}
        if len(digits) == 11 and digits.startswith("1"):
            candidates.add(digits[1:])
        elif len(digits) == 10:
            candidates.add("1" + digits)

        matched_rule = (
            db.execute(
                select(InboundRule).where(
                    InboundRule.rule_set_id == rule_set_id,
                    InboundRule.tenant_id == tenant_id,
                    InboundRule.is_deleted.is_(False),
                    InboundRule.action == "deny",
                    InboundRule.normalized_digits.in_(list(candidates)),
                )
            )
            .scalars()
            .first()
        )

        if matched_rule:
            return True, matched_rule
        return False, None

    def _get_rule_set_or_404(
        self, db: Session, tenant_id: uuid.UUID, rule_set_id: uuid.UUID
    ) -> InboundRuleSet:
        rs = (
            db.execute(
                select(InboundRuleSet).where(
                    InboundRuleSet.id == rule_set_id,
                    InboundRuleSet.tenant_id == tenant_id,
                    InboundRuleSet.is_deleted.is_(False),
                )
            )
            .scalars()
            .first()
        )
        if not rs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inbound rule set {rule_set_id} not found",
            )
        return rs

    def _to_rule_set_response(
        self, db: Session, rs: InboundRuleSet
    ) -> InboundRuleSetResponse:
        rules = (
            db.execute(
                select(InboundRule)
                .where(
                    InboundRule.rule_set_id == rs.id,
                    InboundRule.tenant_id == rs.tenant_id,
                    InboundRule.is_deleted.is_(False),
                )
                .order_by(InboundRule.created_at.desc())
            )
            .scalars()
            .all()
        )
        rule_items = [
            InboundRuleItem(
                id=r.id,
                phone_number_pattern=r.phone_number_pattern,
                normalized_digits=r.normalized_digits,
                label=r.label,
                action=r.action,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in rules
        ]
        return InboundRuleSetResponse(
            id=rs.id,
            name=rs.name,
            description=rs.description,
            rules_count=len(rule_items),
            rules=rule_items,
            created_at=rs.created_at,
            updated_at=rs.updated_at,
        )


inbound_rules_service = InboundRulesService()
