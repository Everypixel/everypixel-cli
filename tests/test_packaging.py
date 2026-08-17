from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


def test_direct_runtime_dependencies_are_declared() -> None:
    root = Path(__file__).parents[1]
    pyproject_path = root / "pyproject.toml"
    package_metadata = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = package_metadata["project"]["dependencies"]
    declared = {
        re.split(r"[<>=!~; \[]", dependency, maxsplit=1)[0].replace("-", "_")
        for dependency in dependencies
    }
    imported: set[str] = set()
    for path in (root / "src" / "everypixel_cli").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

    optional_imports = {"jq"}
    third_party = (
        imported
        - sys.stdlib_module_names
        - {
            "everypixel_cli",
            *optional_imports,
        }
    )
    assert third_party <= declared
