In my_first_poetry_project do:

1. `poetry install` (to install everything needed for this task)
2. `poetry shell` (to get in the environment where you can work with all the tools)
3. If `poetry shell` does not exist, add it with `pip install poetry-plugin-shell` or `poetry self add poetry-plugin-shell`.

Step 1 is only needed for the first time.

Use pytest to test the testcases:
- `pytest .`

Use tools to check styleguide:
- `black . --diff` or `black .` to fix directly
- `ruff . --no-fix --no-cache` or `ruff . --no-cache` to fix directly
- `mypy .`