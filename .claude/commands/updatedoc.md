## Task: Sync documentation files with current repo state

1. **Audit the repo**: Scan all source files, configs, and directories to build a current picture of the project's actual state (features, architecture, dependencies, pipeline logic, models, etc.).

2. **Compare & update each doc file** against the repo's actual state:
   - `CLAUDE.md`
   - `README.md`
   - `explanation.md`
   - `report.md`

   For each file:
   - Identify any outdated, missing, or inaccurate information vs. the current codebase.
   - Add new sections/details for any features, modules, or changes not yet documented.
   - Remove or correct references to code/features that no longer exist.

3. **Normalize the level of detail across all four files**:
   - Determine which file currently has the most comprehensive and granular coverage.
   - Use that file as the detail baseline and bring the others UP to the same depth.
   - Ensure consistent terminology, structure names, and feature descriptions across all files.
   - Each file should still serve its own purpose (CLAUDE.md = dev context, README.md = project overview, explanation.md = technical deep-dive, report.md = project report), but none should be significantly less detailed than the others.

4. **Show a summary of changes** made to each file before committing.