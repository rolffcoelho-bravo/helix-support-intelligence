# A4.4d pre-execution audit

Status: `PASSED_PRE_EXECUTION_VALIDATION_ONLY_NO_RESULTS`

A4.4d is registered as a one-shot validation-only execution. Before the main-branch execution, no validation case is materialized and no semantic inference is performed.

The frozen A4.4a partition contains 20 validation intents. The registered case construction implies 144 cases without opening them: 140 base cases, two archived stale-evidence fixtures, and two unresolved-conflict fixtures. The same registered combinatorics imply 246 eligible semantic pairs with 106 ENTAILED, 20 CONTRADICTED, and 120 UNKNOWN gold relations.

The verifier remains `FacebookAI/roberta-large-mnli` at revision `2a8f12d27941090092df78e4ba6f0928eb5eac98` and the A4.4c temperature remains frozen at `T = 3.67`.

The execution workflow enforces parent SHA `57bb2c81ab2cc2d5b8c1a4928c2600b4a770d110`, so a later edit to the workflow cannot silently rerun the validation experiment. Scientific failure is an admissible terminal result and does not authorize refitting, threshold search, model substitution, candidate scoring, confirmatory access, or post-result rescue.
