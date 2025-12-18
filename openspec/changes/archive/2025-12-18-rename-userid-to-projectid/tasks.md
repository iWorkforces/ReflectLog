## 1. OpenSpec Proposal
- [ ] 1.1 Create proposal.md
- [ ] 1.2 Create design.md
- [ ] 1.3 Create spec delta for message-storage
- [ ] 1.4 Validate with `openspec validate rename-userid-to-projectid --strict`

## 2. Infrastructure Layer
- [ ] 2.1 Update MessageStore (schema, migration, methods)
- [ ] 2.2 Update TantivyEngine (schema, rebuild, methods)
- [ ] 2.3 Update USearchEngine (methods)

## 3. Application Layer
- [ ] 3.1 Update ISemanticSearchEngine protocol in types.py
- [ ] 3.2 Update SearchEngine protocol in protocols.py
- [ ] 3.3 Update MemoryManager calls

## 4. Tests
- [ ] 4.1 Update infrastructure unit tests
- [ ] 4.2 Update application unit tests
- [ ] 4.3 Update integration tests
- [ ] 4.4 Add migration-specific tests

## 5. Documentation
- [ ] 5.1 Update CLAUDE.md files
- [ ] 5.2 Verify all tests pass

## 6. Validation
- [ ] 6.1 Run type check: `./start-type-check.sh`
- [ ] 6.2 Run linting: `./start-lint.sh --all`
- [ ] 6.3 Run tests: `./start-unittest.sh`
