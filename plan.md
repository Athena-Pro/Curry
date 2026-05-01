1. **Update SKILL.md:**
   - Add a note to the `config.json` schema documentation specifying what happens if `core_db` is missing or unreachable at session startup.
   - Add a placeholder note for the `CURRENCY` type format (e.g., `"USD 12.50"`).
   - Document the retry policy in the REU contract (e.g., specifying a default number of retries like 3).
   - Clarify the `export_schema()` outstanding item to note whether it's intended for migration tooling or introspection tooling.

2. **Complete Pre-commit Steps:**
   - Follow instructions from `pre_commit_instructions`.

3. **Submit Change:**
   - Call `submit` to push and merge the changes.
