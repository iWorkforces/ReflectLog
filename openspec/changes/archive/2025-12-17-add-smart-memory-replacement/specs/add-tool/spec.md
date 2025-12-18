# add-tool Spec Delta

## ADDED Requirements

### Requirement: Smart Memory Replacement

The add tool SHALL detect when a new memory semantically replaces an existing memory and automatically remove the old one before storing the new one.

#### Scenario: Replacement detected and executed

- **WHEN** a new memory is added that updates or contradicts an existing memory
- **AND** the LLM confidence score meets the threshold (default: 0.7)
- **THEN** the old memory SHALL be deleted before adding the new one
- **AND** detailed logs SHALL show old memory preview, new memory preview, confidence score, and reason

#### Scenario: No replacement needed

- **WHEN** a new memory is added that does not semantically replace any existing memory
- **THEN** the memory SHALL be added normally without any deletion

#### Scenario: Smart replacement disabled

- **WHEN** `ENABLE_SMART_REPLACE=false` is configured
- **THEN** no replacement detection SHALL occur
- **AND** memories SHALL be added normally with existing deduplication still applied

#### Scenario: LLM failure graceful degradation

- **WHEN** the LLM call for replacement detection fails
- **THEN** a warning SHALL be logged with the error details
- **AND** the memory SHALL be added normally without replacement

### Requirement: Smart Replacement Configuration

The smart memory replacement feature SHALL be configurable via environment variables.

#### Scenario: Default configuration

- **WHEN** no smart replacement environment variables are set
- **THEN** `ENABLE_SMART_REPLACE` SHALL default to `true`
- **AND** `SMART_REPLACE_THRESHOLD` SHALL default to `0.7`

#### Scenario: Disable smart replacement

- **WHEN** `ENABLE_SMART_REPLACE=false` is set
- **THEN** the SmartReplacer component SHALL NOT be initialized
- **AND** no LLM calls for replacement detection SHALL be made

#### Scenario: Custom threshold

- **WHEN** `SMART_REPLACE_THRESHOLD` is set to a value between 0.0 and 1.0
- **THEN** the system SHALL use that threshold for replacement decisions
- **AND** only replacements with confidence >= threshold SHALL be executed

### Requirement: Smart Replacement Logging

The smart memory replacement feature SHALL provide detailed logging for transparency and debugging.

#### Scenario: Replacement executed logging

- **WHEN** a replacement is detected and executed
- **THEN** the log SHALL include the old memory preview (truncated to 80 chars)
- **AND** the log SHALL include the new memory preview (truncated to 80 chars)
- **AND** the log SHALL include the LLM confidence score
- **AND** the log SHALL include the LLM reason for the decision

#### Scenario: No replacement logging

- **WHEN** a replacement check is performed but no replacement is needed
- **THEN** the system SHALL log at DEBUG level with confidence score and reason
