"""Resolve immutable application and workspace storage paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from platformdirs import PlatformDirs

from .. import constants


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Describe storage belonging to one initialized workspace.

    Args:
        data (Path): UUID-scoped durable workspace data directory.
        configuration (Path): Project-local sparse configuration override.
        sessions (Path): Central session database.
        permissions (Path): Central permission policy.
    """

    data: Path
    configuration: Path
    sessions: Path
    permissions: Path


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    """Describe immutable platform-native application storage.

    Args:
        configuration_root (Path): User-editable configuration directory.
        data_root (Path): Durable application data directory.
        state_root (Path): Operational application state directory.
    """

    configuration_root: Path
    data_root: Path
    state_root: Path

    def __post_init__(self) -> None:
        for name in ("configuration_root", "data_root", "state_root"):
            path = Path(getattr(self, name)).expanduser()
            if not path.is_absolute():
                raise ValueError(f"Application {name.removesuffix('_root')} root must be absolute.")
            object.__setattr__(self, name, path.resolve())

    @classmethod
    def discover(cls, environment: Mapping[str, str] | None = None) -> ApplicationPaths:
        """Discover paths from explicit overrides and platform conventions.

        Args:
            environment (Mapping[str, str] | None): Environment source. Defaults to
                :data:`os.environ`.

        Returns:
            ApplicationPaths: Validated immutable application paths.

        Raises:
            ValueError: If an explicit override is relative.
        """
        values = os.environ if environment is None else environment
        directories = PlatformDirs(constants.APP_NAME, appauthor=False, roaming=False)
        return cls(
            cls._root(values, "LOOP_CONFIG_HOME", directories.user_config_path),
            cls._root(values, "LOOP_DATA_HOME", directories.user_data_path),
            cls._root(values, "LOOP_STATE_HOME", directories.user_state_path),
        )

    @staticmethod
    def _root(environment: Mapping[str, str], variable: str, default: Path) -> Path:
        """Return one absolute environment override or its platform default."""
        value = environment.get(variable)
        if value is None or not value.strip():
            return Path(default)
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{variable} must be an absolute path.")
        return path

    @property
    def user_configuration(self) -> Path:
        """Return the user configuration file.

        Returns:
            Path: Complete user configuration file.
        """
        return self.configuration_root / constants.APP_CONFIGURATION_FILENAME

    @property
    def user_permissions(self) -> Path:
        """Return the user-wide remembered-permission policy path.

        Returns:
            Path: User-owned YAML policy applied in every workspace.
        """
        return self.configuration_root / constants.PERMISSIONS_FILENAME

    @property
    def workspace_catalog(self) -> Path:
        """Return the workspace registry database.

        Returns:
            Path: Workspace registry database.
        """
        return self.data_root / constants.WORKSPACE_DATABASE_FILENAME

    @property
    def telemetry(self) -> Path:
        """Return the telemetry database.

        Returns:
            Path: Global telemetry database.
        """
        return self.state_root / constants.TELEMETRY_DATABASE_FILENAME

    @property
    def permissions_audit(self) -> Path:
        """Return the permission audit database.

        Returns:
            Path: Global permission audit database.
        """
        return self.state_root / constants.PERMISSIONS_AUDIT_FILENAME

    @property
    def operational_log(self) -> Path:
        """Return the operational log.

        Returns:
            Path: Global operational log.
        """
        return self.state_root / constants.OPERATIONAL_LOG_FILENAME

    def for_workspace(self, workspace_id: str, root: Path | str) -> WorkspacePaths:
        """Derive immutable storage paths for an initialized workspace.

        Args:
            workspace_id (str): Durable identifier used as one safe path segment.
            root (Path | str): Canonical workspace root containing local overrides.

        Returns:
            WorkspacePaths: Paths owned by the identified workspace.

        Raises:
            ValueError: If the identifier is empty or path-like.
        """
        if not workspace_id or Path(workspace_id).name != workspace_id:
            raise ValueError("Workspace identifier must be a non-empty path segment.")
        data = self.data_root / "workspaces" / workspace_id
        return WorkspacePaths(
            data=data,
            configuration=Path(root).resolve()
            / constants.APP_DIRECTORY
            / constants.APP_CONFIGURATION_FILENAME,
            sessions=data / constants.SESSION_DATABASE_FILENAME,
            permissions=data / constants.PERMISSIONS_FILENAME,
        )
