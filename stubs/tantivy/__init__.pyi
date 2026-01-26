"""Type stubs for tantivy-py library."""

from typing import Any

class FieldEntry:
    """A field entry in the schema."""

    ...

class Schema:
    """Tantivy schema defining the document structure."""

    def get_field(self, field_name: str) -> FieldEntry:
        """Get a field by name. Raises ValueError if not found."""
        ...

class SchemaBuilder:
    """Builder for constructing a Tantivy schema."""

    def add_text_field(
        self,
        name: str,
        *,
        stored: bool = True,
        tokenizer_name: str = "default",
        index_option: str = "position",
    ) -> FieldEntry:
        """Add a text field to the schema."""
        ...

    def add_unsigned_field(
        self,
        name: str,
        *,
        stored: bool = True,
        indexed: bool = False,
        fast: bool = False,
    ) -> FieldEntry:
        """Add an unsigned integer field to the schema."""
        ...

    def add_integer_field(
        self,
        name: str,
        *,
        stored: bool = True,
        indexed: bool = False,
        fast: bool = False,
    ) -> FieldEntry:
        """Add a signed integer field to the schema."""
        ...

    def add_date_field(
        self,
        name: str,
        *,
        stored: bool = True,
        indexed: bool = False,
        fast: bool = False,
    ) -> FieldEntry:
        """Add a date field to the schema."""
        ...

    def build(self) -> Schema:
        """Build the schema."""
        ...

class Document:
    """A Tantivy document."""

    def add_text(self, field: str, value: str) -> None:
        """Add a text value to the document."""
        ...

    def add_unsigned(self, field: str, value: int) -> None:
        """Add an unsigned integer value to the document."""
        ...

    def add_integer(self, field: str, value: int) -> None:
        """Add a signed integer value to the document."""
        ...

    def get_first(self, field: str) -> Any | None:
        """Get the first value for a field."""
        ...

class TopDocs:
    """Search results from a Tantivy query."""

    hits: list[tuple[float, Any]]

class Searcher:
    """A Tantivy searcher for executing queries."""

    def search(self, *, query: Query, limit: int) -> TopDocs:
        """Execute a search query."""
        ...

    def doc(self, doc_address: Any) -> Document:
        """Retrieve a document by address."""
        ...

    def num_docs(self) -> int:
        """Return the total number of documents in the index."""
        ...

class IndexWriter:
    """A Tantivy index writer for adding/deleting documents."""

    def add_document(self, doc: Document) -> int:
        """Add a document to the index."""
        ...

    def delete_documents(self, *, field: str, text: str) -> None:
        """Delete documents matching field value."""
        ...

    def commit(self) -> None:
        """Commit pending changes."""
        ...

    def wait_merging_threads(self) -> None:
        """Wait for background merge threads to complete."""
        ...

class Query:
    """A Tantivy query."""

    ...

class Index:
    """A Tantivy index for full-text search."""

    schema: Schema

    def __init__(
        self,
        schema: Schema,
        *,
        path: str | None = None,
        reuse: bool = False,
    ) -> None:
        """Create a new index with the given schema."""
        ...

    @staticmethod
    def open(path: str) -> Index:
        """Open an existing index from disk."""
        ...

    def writer(self, heap_size: int = ...) -> IndexWriter:
        """Get an index writer."""
        ...

    def searcher(self) -> Searcher:
        """Get a searcher for the index."""
        ...

    def parse_query(
        self,
        query: str,
        *,
        default_field_names: list[str] | None = None,
    ) -> Query:
        """Parse a query string."""
        ...

    def reload(self) -> None:
        """Reload the index to see recent changes."""
        ...
