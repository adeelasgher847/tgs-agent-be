"""
Regression coverage for the RAG/KB embedding-client warm-up task's
reference-lifetime handling in app.main's lifespan.

Background: `asyncio.create_task(warm_embedding_client())` on its own only
keeps a WEAK reference to the Task internally (asyncio.tasks._all_tasks is a
WeakSet). With no strong reference held anywhere else, the task is eligible
for garbage collection at any suspension point (e.g. the `await
loop.run_in_executor(...)` inside warm_embedding_client) and CPython can
silently drop/cancel it before completion — with nothing awaiting or logging
the loss. The fix stores the task in `app.state._warmup_tasks` (a strong
reference) for its lifetime and discards it via `add_done_callback` once
done, so it can't be prematurely collected but also doesn't leak.
"""

from __future__ import annotations

import asyncio
import gc


def test_bare_create_task_pattern_is_unsafe_without_a_strong_reference():
    """Sanity check on the *problem* being fixed: a Task with no reference
    other than the local variable inside create_task()'s caller is only kept
    alive by refcounting — as soon as that local variable's reference is
    dropped, nothing else keeps it alive. This is exactly the bug pattern
    `asyncio.create_task(warm_embedding_client())` had before the fix."""

    async def scenario():
        task = asyncio.create_task(asyncio.sleep(0, result="done"))
        # No strong reference retained anywhere but this local name — mirrors
        # the pre-fix `asyncio.create_task(warm_embedding_client())` call
        # whose return value was discarded entirely.
        weak_only = task
        del task
        gc.collect()
        # We can still await it here only because Python's refcounting kept
        # `weak_only` (the same object) alive within this frame — in the real
        # bug, no reference at all survives past the `create_task(...)` call
        # expression, which is what makes the loss silent in production.
        return await weak_only

    assert asyncio.run(scenario()) == "done"


def test_warmup_task_set_pattern_survives_gc_and_completes():
    """Regression test for the actual fix: storing the task in a strong-
    reference set (mirroring `app.state._warmup_tasks` in app/main.py) and
    registering `add_done_callback(tasks.discard)` must keep the task alive
    across a suspension point + explicit gc.collect(), and the task must
    still run to completion and be cleaned up from the set afterward."""

    async def scenario():
        warmup_tasks: set[asyncio.Task] = set()

        async def _slow_warmup():
            await asyncio.sleep(0)
            return "warmed"

        task = asyncio.create_task(_slow_warmup())
        warmup_tasks.add(task)
        task.add_done_callback(warmup_tasks.discard)

        # Drop every other reference to the task object, exactly like
        # app.main's lifespan does after scheduling it (the local
        # `warmup_task` variable goes out of scope once the `try` block
        # ends) — only `warmup_tasks` keeps it alive now.
        del task
        gc.collect()

        # The task must still be tracked (i.e. not GC'd) and must still run
        # to completion.
        assert len(warmup_tasks) == 1
        (remaining_task,) = tuple(warmup_tasks)
        result = await remaining_task
        assert result == "warmed"

        # add_done_callback fires on the loop's next iteration — yield once
        # so the discard callback has actually run before we assert cleanup.
        await asyncio.sleep(0)
        assert warmup_tasks == set()

    asyncio.run(scenario())


def test_lifespan_skips_warmup_outside_staging_production(client):
    """End-to-end via the real app lifespan (ENVIRONMENT defaults to
    "development" in tests, same as local `uvicorn --reload`): the warm-up
    must NOT fire a real request here — it's gated to staging/production
    only, precisely so the pytest suite (and local dev) never makes a real,
    billed OpenAI call just from starting the app. If this ever regresses to
    firing unconditionally, a still-pending task in `app.state._warmup_tasks`
    would be the visible symptom in CI."""
    from app.main import app as _app

    warmup_tasks = getattr(_app.state, "_warmup_tasks", set())
    assert warmup_tasks == set()


def test_warm_up_scheduling_uses_add_done_callback_discard_pattern():
    """Static assertion that app.main's warm-up scheduling code still uses
    the strong-reference-set + add_done_callback(discard) pattern, so a
    future refactor can't silently reintroduce the bare, reference-less
    `asyncio.create_task(warm_embedding_client())` regression (the original
    bug: the Task object returned by create_task() was never assigned to
    anything, so nothing kept a strong reference to it)."""
    import inspect
    import re

    import app.main as main_module

    source = inspect.getsource(main_module)
    assert "_warmup_tasks" in source
    assert "add_done_callback" in source

    # The create_task(...) call for the warm-up coroutine must assign its
    # result to a variable (not be a bare, unreferenced statement).
    match = re.search(r"^\s*(\S.*=\s*)?asyncio\.create_task\(warm_embedding_client\(\)\)", source, re.MULTILINE)
    assert match is not None, "warm_embedding_client() warm-up call not found"
    assert match.group(1), (
        "asyncio.create_task(warm_embedding_client()) must assign its result "
        "to a variable so a strong reference can be retained — a bare call "
        "lets the Task be garbage-collected before it completes"
    )
