"""Read and write YAML using the application's single YAML library."""

from io import StringIO

from ruamel.yaml import YAML, YAMLError
from ruamel.yaml.constructor import DuplicateKeyError


def load_yaml(content: str | bytes) -> object:
    """Parse a YAML document and reject ambiguous duplicate keys.

    Use YAML 1.2 unless the document declares another supported version.

    Args:
        content (str | bytes): YAML document text or UTF-8 encoded bytes.

    Returns:
        object: Parsed document, or None for an empty document.

    Raises:
        YAMLError: The document is invalid, ambiguous, or too deeply nested.
    """
    try:
        parser = YAML(typ="rt", pure=True)
        parser.version = (1, 2)
        document = parser.compose(content)
        if document is None:
            return None
        # Round-trip construction checks duplicate keys inside inline merges. Safe
        # construction preserves recursive aliases, which round-trip loading loses.
        parser.constructor.construct_document(document)
        safe = YAML(typ="safe", pure=True)
        safe.version = parser.version
        return safe.constructor.construct_document(document)
    except YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        position = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(error, "problem", None)
        category = (
            "duplicate key"
            if isinstance(error, DuplicateKeyError)
            else "unhashable key"
            if problem == "found unhashable key"
            else type(error).__name__
        )
        raise YAMLError(f"Invalid YAML ({category}){position}.") from None
    except RecursionError:
        raise YAMLError("YAML nesting exceeds the supported depth.") from None
    except (ValueError, TypeError, OverflowError) as error:
        raise YAMLError(f"Invalid YAML scalar ({type(error).__name__}).") from None


def dump_yaml(value: object) -> str:
    """Serialize a value as YAML 1.2 while preserving mapping order.

    Args:
        value (object): Value to serialize.

    Returns:
        str: YAML document text.

    Raises:
        YAMLError: The value cannot be serialized.
    """
    writer = YAML(typ="rt", pure=True)
    writer.version = (1, 2)
    writer.default_flow_style = False
    writer.sort_base_mapping_type_on_output = False
    stream = StringIO()
    writer.dump(value, stream)
    return stream.getvalue()
