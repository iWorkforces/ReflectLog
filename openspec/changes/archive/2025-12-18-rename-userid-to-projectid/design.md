## Context

The `user_id` field was named during early development but has always functioned as a project-level identifier. The MemoryManager always passes `self.project_id` as the `user_id` parameter. This naming inconsistency causes confusion when reading the codebase.

## Goals / Non-Goals

**Goals:**
- Rename `user_id` to `project_id` throughout the codebase
- Preserve all existing data through auto-migration
- Maintain backward compatibility with existing database files

**Non-Goals:**
- Adding multi-user support (out of scope)
- Changing the functionality of the field (pure rename)

## Decisions

### Decision: libSQL Migration Strategy
Use `ALTER TABLE RENAME COLUMN` (supported since SQLite 3.25.0/libSQL) for in-place column rename. Recreate indices with new names.

**Why:** Atomic, preserves data, no temporary tables needed.

### Decision: Tantivy Migration Strategy
Delete old index and rebuild from libSQL (source of truth). Keep backup until verified.

**Why:** Tantivy does not support field renaming. Since libSQL is the source of truth, rebuilding is safe.

### Decision: Schema Detection
Check for old schema via `pragma_table_info` (libSQL) and field name inspection (Tantivy). Migrate automatically on first access.

**Why:** Seamless upgrade path for users with existing data.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Data loss during Tantivy rebuild | Keep backup, rebuild from libSQL source of truth |
| Migration failure | Log errors, allow retry, backup restoration |
| Performance impact on first load | One-time cost, acceptable for data integrity |

## Migration Plan

1. **Detection Phase**: Check schema version on initialization
2. **libSQL Migration**: Rename column and indices
3. **Tantivy Migration**: Delete index, rebuild from libSQL
4. **Verification**: Confirm data integrity post-migration
5. **Rollback**: Restore from backup if verification fails

## Open Questions

None - approach is straightforward given libSQL is the source of truth.
