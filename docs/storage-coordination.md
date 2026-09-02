# Storage coordination

ReflectLog coordinates workspace writes on the **local filesystem** with
`portalocker==4.3.0`. There is no Redis or distributed lock, and no ACID
claim across SQLite, USearch, and Tantivy.

## Sidecars

Each workspace root is `indexes/<workspace_id.lower()>/` and contains:

- `.reflectlog.writer.lock` — exclusive/shared Portalocker lease
- `.reflectlog.storage-generation` — integer generation published after
  SQLite, USearch, and enabled Tantivy converge

NFS client-local locks and SMB/CIFS `nobrl` are **unsupported**.

## Engines

- **USearch** publishes HNSW snapshots via a same-directory temp file,
  validate, fsync, and `os.replace`. It does not publish generation.
- **Tantivy** scopes readers (shared) and writers (exclusive) to coordinator
  leases. Request-path delete/compact rewrite in place. Leftover
  `.rebuild-bak` restore is startup-only.

## Shutdown

- POSIX: `SIGINT` and `SIGTERM` persist then exit.
- Windows: `SIGBREAK` via `CTRL_BREAK_EVENT` to a new process group.
- Forced termination is a separate abrupt-death case. Lock files are not
  deleted to recover a live owner.

## Capacity

Supported characterization is under 10,000 records per workspace. Do not
treat unpublished speedup ratios as SLOs.
