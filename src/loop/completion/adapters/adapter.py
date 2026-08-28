"""Define the interactive completion adapter contract."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from prompt_toolkit.document import Document

from ..models import CompletionMatch, CompletionValue


class CompletionAdapter(ABC):
    """Provide candidates for one independently registered completion capability."""

    @property
    def front_markers(self) -> tuple[str, ...]:
        """Return symbols that can activate the adapter.

        Returns:
            tuple[str, ...]: Declared front-marker symbols.
        """
        return ()

    @property
    def keywords(self) -> tuple[str, ...]:
        """Return leading keywords that can activate the adapter.

        Returns:
            tuple[str, ...]: Declared activation keywords.
        """
        return ()

    @abstractmethod
    def match(self, document: Document) -> CompletionMatch | None:
        """Match the input at the cursor and describe its replacement region.

        Args:
            document (Document): Current editable input and cursor position.

        Returns:
            CompletionMatch | None: Active match, or ``None`` when the adapter is inactive.
        """

    @abstractmethod
    def complete(self, match: CompletionMatch) -> Iterable[CompletionValue]:
        """Return candidates for an active match.

        Args:
            match (CompletionMatch): Activation returned by ``match``.

        Returns:
            Iterable[CompletionValue]: Candidates for the active fragment.
        """
