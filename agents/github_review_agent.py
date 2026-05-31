"""
GitHub Code Review Agent for happyassist.ai backend.

Uses Claude claude-opus-4-7 with tool use to:
  - Inspect changed files and diffs
  - Run linting (flake8/pylint) and type checks (mypy)
  - Review code quality, security, and FastAPI/SQLAlchemy standards
  - Produce a structured Markdown report

Usage:
    python agents/github_review_agent.py                   # review staged/unstaged vs HEAD
    python agents/github_review_agent.py --base v1         # review current branch vs v1
    python agents/github_review_agent.py --pr 42           # review GitHub PR #42 (needs gh CLI)
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Optional

import anthropic

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _run(cmd: list[str], cwd: str = ".") -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    out = result.stdout + result.stderr
    return out.strip() if out.strip() else "(no output)"


def list_changed_files(base: str) -> str:
    """Return newline-separated list of files changed vs base ref."""
    return _run(["git", "diff", "--name-only", base, "HEAD"])


def get_git_diff(base: str, file_path: Optional[str] = None) -> str:
    """Return unified diff vs base ref, optionally scoped to one file."""
    cmd = ["git", "diff", base, "HEAD", "--", file_path] if file_path else ["git", "diff", base, "HEAD"]
    diff = _run(cmd)
    # Truncate very large diffs to stay within context limits
    if len(diff) > 20_000:
        diff = diff[:20_000] + "\n... (truncated, diff too large)"
    return diff


def get_file_content(file_path: str) -> str:
    """Read a file from the working tree."""
    try:
        with open(file_path) as f:
            content = f.read()
        if len(content) > 10_000:
            content = content[:10_000] + "\n... (truncated)"
        return content
    except FileNotFoundError:
        return f"File not found: {file_path}"


def run_linter(file_path: Optional[str] = None) -> str:
    """Run flake8 on a file or the whole project."""
    target = file_path or "."
    out = _run(["python", "-m", "flake8", "--max-line-length=100", target])
    return out if out else "No linting issues found."


def run_type_check(file_path: Optional[str] = None) -> str:
    """Run mypy for type checking."""
    target = file_path or "app"
    out = _run(["python", "-m", "mypy", "--ignore-missing-imports", target])
    return out if out else "No type errors found."


def run_tests() -> str:
    """Run the test suite and return a summary."""
    out = _run(["python", "-m", "pytest", "--tb=short", "-q"])
    return out


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "list_changed_files",
        "description": "List all files changed in this branch compared to a base ref.",
        "input_schema": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "Base git ref, e.g. 'main' or 'HEAD~1'"}
            },
            "required": ["base"],
        },
    },
    {
        "name": "get_git_diff",
        "description": "Get the unified git diff for all changes or a specific file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "Base git ref"},
                "file_path": {"type": "string", "description": "Optional: restrict diff to this file"},
            },
            "required": ["base"],
        },
    },
    {
        "name": "get_file_content",
        "description": "Read the full content of a file from the working tree.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path to the file"}
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "run_linter",
        "description": "Run flake8 linting on a file or the whole project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Optional file to lint; omit for full project"}
            },
            "required": [],
        },
    },
    {
        "name": "run_type_check",
        "description": "Run mypy type checking on a file or the app/ directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Optional file or directory to check"}
            },
            "required": [],
        },
    },
    {
        "name": "run_tests",
        "description": "Run the pytest test suite and return the results.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def execute_tool(name: str, inputs: dict[str, Any]) -> str:
    if name == "list_changed_files":
        return list_changed_files(inputs["base"])
    if name == "get_git_diff":
        return get_git_diff(inputs["base"], inputs.get("file_path"))
    if name == "get_file_content":
        return get_file_content(inputs["file_path"])
    if name == "run_linter":
        return run_linter(inputs.get("file_path"))
    if name == "run_type_check":
        return run_type_check(inputs.get("file_path"))
    if name == "run_tests":
        return run_tests()
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior backend code reviewer specializing in Python, FastAPI, SQLAlchemy,
PostgreSQL, and Alembic. You are reviewing a pull-request for the happyassist.ai backend.

Your review covers:
1. **Correctness** – Logic errors, edge cases, data integrity issues.
2. **Security** – SQL injection, insecure deserialization, hardcoded secrets, missing auth checks.
3. **FastAPI / Pydantic standards** – Proper response_model, status codes, dependency injection.
4. **SQLAlchemy patterns** – Session handling, lazy vs eager loading, N+1 queries.
5. **Code quality** – Readability, naming, duplication, unnecessary complexity.
6. **Test coverage** – Whether the changes include adequate tests.
7. **Alembic** – Migration files present and correctly structured for schema changes.

Use the provided tools to inspect the diff, read files, and run automated checks before writing
your final review. Then produce a structured Markdown report with the following sections:

## Summary
## Critical Issues  (security / correctness blockers)
## Warnings         (should-fix but not blockers)
## Suggestions      (nice-to-have improvements)
## Automated Check Results  (linter / type checker / tests)
## Verdict          (APPROVE / REQUEST CHANGES / NEEDS DISCUSSION)
"""


def run_review(base: str) -> None:
    client = anthropic.Anthropic()

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Please review the changes on this branch compared to `{base}`. "
                "Use the available tools to gather the diff, inspect changed files, "
                "run linting and type checks, and execute the test suite. "
                "Then produce a comprehensive Markdown review report."
            ),
        }
    ]

    print(f"\n🔍  Starting code review (base: {base}) …\n{'─' * 60}\n")

    # Agentic loop
    while True:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            cache_control={"type": "ephemeral"},  # cache the stable system + tools prefix
        )

        # Accumulate the assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Print the final review
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  ⚙  {block.name}({json.dumps(block.input, separators=(',', ':'))})")
                    result = execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason – bail out
        print(f"Unexpected stop_reason: {response.stop_reason}")
        break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub code review agent")
    parser.add_argument(
        "--base",
        default="v1",
        help="Base git ref to compare against (default: main)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="GitHub PR number – fetches the base branch via `gh pr view` (requires gh CLI)",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    base = args.base

    if args.pr:
        # Resolve the base branch for the PR using the GitHub CLI
        result = subprocess.run(
            ["gh", "pr", "view", str(args.pr), "--json", "baseRefName", "-q", ".baseRefName"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Could not fetch PR #{args.pr}: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        base = result.stdout.strip()
        print(f"PR #{args.pr} → base branch: {base}")

    run_review(base)


if __name__ == "__main__":
    main()
