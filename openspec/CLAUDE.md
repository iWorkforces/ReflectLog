# openspec/

This directory contains OpenSpec specifications for managing project changes and proposals.

## Structure

```
openspec/
├── AGENTS.md       # Instructions for AI agents on using OpenSpec
├── project.md      # Project metadata and configuration
├── changes/        # Active and archived change proposals
└── specs/          # Shared specification documents
```

## Purpose

OpenSpec is a specification management system for:
- Proposing and tracking project changes
- Documenting architectural decisions
- Managing change history through archival
- Sharing specifications across multiple changes

## Usage

See `openspec/AGENTS.md` for instructions on:
- Creating new change proposals
- Applying changes to the codebase
- Managing the specification lifecycle

## Directory Details

### `changes/`

Contains change proposals:
- Active changes (proposed but not yet applied)
- Archived changes (completed or cancelled)
- Each change has its own subdirectory with specs

### `specs/`

Shared specification documents referenced by multiple changes:
- `search-optimization/` - Search performance and accuracy improvements
- `smart-memory-replacement/` - Memory replacement feature specifications

## Creating a New Change

1. Create a new directory under `changes/`
2. Add a `specs/` subdirectory for change-specific specs
3. Document the change proposal following OpenSpec conventions
4. Reference shared specs from `specs/` when applicable
