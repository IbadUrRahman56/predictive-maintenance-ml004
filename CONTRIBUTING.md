# Contributing — Team Workflow

Two ML engineers, 6-day sprint. Suggested split so you're not blocked on
each other:

- **Engineer A**: data/feature engineering, model training, evaluation
- **Engineer B**: explainability, MLOps/tracking, API, dashboard

## Branching model

```
main        → always deployable, protected
develop     → integration branch, merge here first
feature/*   → one branch per task, e.g. feature/rul-model, feature/api
```

## Day-to-day flow

```bash
git checkout develop
git pull
git checkout -b feature/<short-task-name>

# ...do the work, commit as you go...
git add <files>
git commit -m "feat: <what you did>"

git push -u origin feature/<short-task-name>
# open a Pull Request into develop, tag your teammate as reviewer
```

## Commit message convention

```
feat: new functionality
fix: bug fix
docs: documentation only
test: adding/adjusting tests
refactor: code change that doesn't change behavior
chore: tooling, CI, deps
```

## Before opening a PR

- [ ] `pytest tests/` passes locally
- [ ] `python src/feature_engineering.py` and `python src/train_models.py` run clean
- [ ] No large generated files committed (`models/*.joblib`, `mlflow.db`, `data/*.parquet` are gitignored — don't force-add them)
- [ ] README updated if you changed how something runs

## End of sprint

Merge `develop` → `main`, tag the release:

```bash
git checkout main
git merge develop
git tag -a v1.0 -m "ML-004 Predictive Maintenance Platform — sprint 1"
git push origin main --tags
```
