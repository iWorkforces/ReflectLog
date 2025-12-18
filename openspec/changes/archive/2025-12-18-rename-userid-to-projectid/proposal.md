# Change: Rename user_id to project_id

## Why

The `user_id` field is misleading - it stores project identifiers, not user identifiers. Internally, it's always set to `self.project_id` in MemoryManager. This rename improves code clarity and semantic accuracy.

## What Changes

- Rename `user_id` to `project_id` in MessageRecord dataclass
- Rename database column `user_id` to `project_id` and update indices
- Rename Tantivy schema field (requires index rebuild)
- Update all method signatures across protocols and implementations
- Add auto-migration for existing data (libSQL column rename, Tantivy index rebuild)

## Impact

- Affected specs: message-storage
- Affected code: infrastructure layer (3 files), application layer (3 files), tests (5 files)
- **BREAKING**: Existing indexes will be auto-migrated on first access
