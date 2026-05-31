"""
Backfill script: ingest all active agents' system prompts into Pinecone RAG.

Use this one-time after enabling auto-ingest so existing agents become RAG-ready.

Run from project root:
    ./venv/bin/python -m scripts.backfill_agent_prompt_kb
"""

from __future__ import annotations

from app.db.session import SessionLocal
from app.models.agent import Agent
from app.services.agent_service import agent_service


def main() -> None:
    db = SessionLocal()
    try:
        agents = (
            db.query(Agent)
            .filter(Agent.is_deleted == False)  # noqa: E712
            .all()
        )
        if not agents:
            print("[INFO] No active agents found.")
            return

        total = len(agents)
        attempted = 0
        skipped = 0

        print(f"[INFO] Found {total} active agent(s). Starting backfill...")
        for agent in agents:
            prompt_text = (agent.system_prompt or "").strip()
            if not prompt_text:
                skipped += 1
                print(f"[SKIP] agent_id={agent.id} name={agent.name!r} (empty system_prompt)")
                continue

            attempted += 1
            try:
                agent_service.ensure_agent_prompt_ingested(db, agent)
                print(f"[OK] agent_id={agent.id} name={agent.name!r}")
            except Exception as e:
                print(f"[FAIL] agent_id={agent.id} name={agent.name!r} error={e}")

        print("\n=== Backfill Summary ===")
        print(f"total_agents={total}")
        print(f"attempted={attempted}")
        print(f"skipped_empty_prompt={skipped}")
        print("done")
    finally:
        db.close()


if __name__ == "__main__":
    main()
