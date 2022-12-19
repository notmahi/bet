# Internal Developer Guide

## Pre-commit

```bash
black .
```

## Create model weights archive

```bash
tar -czvf train_runs.tar.gz \
./train_runs/train_kitchen/reproduction \
./train_runs/train_blockpush/reproduction
```

## Branching Convention

- `original-paper`: tracks the original codebase. Read-only.
- `main`: the main branch of this repository.
    - PR protected. Linear history only.
- `<you-name>/<feature-name>`: for new features.

## Example Workflows

### Starting a new feature

- Update your main:
    - `git checkout main`
    - `git pull`
- Create your feature branch:
    - `git checkout -b your-name/your-feature`

### Working on your feature

- Update your main:
    - `git checkout main`
    - `git pull`
- Checkout your feature branch:
    - `git checkout your-name/your-feature`
- Rebase from main (do it frequently to resolve conflicts early):
    - `git rebase main`
- Do some changes.
- Format your code and check the tests.
- Commit your changes:
    - `git add <files>`
    - `git commit -m "Some meaningful and grammatically correct message."`

### Sharing your feature:

- Update your main:
    - `git checkout main`
    - `git pull`
- Checkout your feature branch:
    - `git checkout your-name/your-feature`
- Rebase from main:
    - `git rebase main`
- Push your branch:
    - `git push`
- Create a PR on GitHub from your branch to `main`.