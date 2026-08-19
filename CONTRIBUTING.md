# Contributing

Helix welcomes focused contributions that improve the documented public product scope without expanding it implicitly.

## Development

```bash
make setup
make quality
```

Use Python 3.12+, typed interfaces, deterministic tests, and the `src/` package layout. Exploratory notebooks cannot define production behaviour.

## Change requirements

Every contribution should:

1. state the problem and public value;
2. remain within the documented product contract;
3. add or update tests for changed behaviour;
4. pass `make quality`;
5. update an ADR when it changes an architectural boundary;
6. avoid unsupported performance or impact claims.

## Public-material review

Before committing, confirm that the change contains no secrets, personal data, private prompts, internal planning, unpublished commercial strategy, workstation paths, research-only notes, or confidential security detail. Run:

```bash
make publication-audit
```

If material is not intended for public scrutiny, keep it outside this repository.

## Pull requests

Keep pull requests focused and explain what changed, why it matters, how it was validated, and which limitations remain. Contributions may be declined when they add complexity without measurable value or exceed the documented public scope.
