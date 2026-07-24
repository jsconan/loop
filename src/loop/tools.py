"""Tools that can be called from the loop."""

import json
from dataclasses import dataclass
from datetime import datetime
from warnings import warn

TOOLS = [
    {
        "type": "function",
        "name": "read_text_file",
        "description": "Read the contents of a text file from the local disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the text file to read.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "write_text_file",
        "description": "Write content to a text file on the local disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the text file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_current_datetime",
        "description": "Return the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def read_text_file(path: str) -> str:
    """Read the contents of a text file and return it as a string."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            if not content:
                return f"File '{path}' is empty."
            return content
    except Exception as e:  # pylint: disable=broad-except
        return f"Error reading file: {e}"


def write_text_file(path: str, content: str) -> str:
    """Write the given content to a text file."""
    if input(f"Agent wants to write to file '{path}'. Proceed? [y/N]: ").strip().lower() != "y":
        return "Write operation cancelled."

    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Successfully wrote to file '{path}'."
    except Exception as e:  # pylint: disable=broad-except
        return f"Error writing to file: {e}"


def get_current_datetime() -> str:
    """Return the current date and time as a string."""

    return datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")


TOOL_FUNCTIONS = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "get_current_datetime": get_current_datetime,
}


@dataclass
class ToolCall:
    """A record of a tool call made by the LLM during reasoning."""

    name: str
    arguments: str

    def call(self) -> str:
        """Call the tool function with the provided arguments and return the result."""
        if self.name not in TOOL_FUNCTIONS:
            return f"Error: Tool '{self.name}' is not available."

        try:
            return TOOL_FUNCTIONS[self.name](**self.decode(self.arguments))
        except Exception as e:  # pylint: disable=broad-except
            return f"Error calling tool '{self.name}': {e}"

    @staticmethod
    def decode(arguments: str) -> dict:
        """Decode a JSON string of tool call arguments into a dictionary."""
        if not arguments or arguments.strip() == "{}":
            return {}

        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            warn(
                f"Failed to decode tool call arguments: {arguments}",
                stacklevel=2,
            )
            return {}
