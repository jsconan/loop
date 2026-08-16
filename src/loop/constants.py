"""Define shared application constants."""

from pathlib import Path

# Application home directory, used for storing session data and other application state.
APP_DIRECTORY = Path(".loop")

# Marker file for identifying the root of a Loop project.
GIT_DIRECTORY = Path(".git")

# Session-related constants
SESSION_DATABASE_FILENAME = "sessions.db"
DEFAULT_SESSION_NAME = "Untitled session"
INITIAL_SESSION_NAME_MAX_CHARS = 48
SESSION_NAME_MAX_CHARS = 80
SESSION_TITLE_CONTEXT_MAX_CHARS = 500
SESSION_TITLE_MAX_WORDS = 6
DEFAULT_COMPACTION_THRESHOLD = 0.8

# Permissions-related constants
PERMISSIONS_FILENAME = "permissions.yaml"
PERMISSIONS_AUDIT_FILENAME = "permissions-audit.jsonl"

# Ignore files
AGENT_IGNORE_FILENAME = ".agentignore"
GIT_IGNORE_FILENAME = ".gitignore"
IGNORE_FILENAMES = (GIT_IGNORE_FILENAME, AGENT_IGNORE_FILENAME)

# Default filenames and directories for agents and skills
DEFAULT_AGENTS_FILENAME = "AGENTS.md"
DEFAULT_SKILL_FILENAME = "SKILL.md"
DEFAULT_SKILLS_DIRECTORY = Path(".agents/skills")
RESOURCE_DIRECTORIES = ("references", "scripts", "assets")
TRUNCATION_MARKER = "\n\n[AGENTS.md truncated: instruction byte limit reached.]"

# Constants for content size limits
MAX_AGENTS_BYTES = 32 * 1024
MAX_CATALOG_CHARS = 8_000
MAX_INSTRUCTIONS_BYTES = 64 * 1024
MAX_TOOL_RESULT_BYTES = 20 * 1024
MAX_TOOL_CONTENT_BYTES = MAX_TOOL_RESULT_BYTES * 4 // 5
MAX_FETCH_BYTES = 10 * 1024 * 1024
MAX_OUTPUT_CHARS = 1_000_000
CONTENT_PREVIEW_MAX_CHARS = 2_000
CONTENT_PREVIEW_MAX_LINES = 20
TOOL_CALL_VALUE_MAX_CHARS = 20

# UI constants
TABULAR_MAX_WIDTH = 120

# Command execution timeout in seconds
COMMAND_TIMEOUT_SECONDS = 30
