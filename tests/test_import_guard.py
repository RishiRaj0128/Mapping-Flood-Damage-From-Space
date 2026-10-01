"""Enforces architectural isolation: src/floodmap must NEVER import evaluation."""

import ast
from pathlib import Path


def test_main_package_never_imports_evaluation():
    """Scans all Python source files in src/ to ensure zero imports of evaluation."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    assert src_dir.exists(), f"Source directory {src_dir} not found!"

    violations = []

    for py_file in src_dir.rglob("*.py"):
        try:
            with open(py_file, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
        except Exception as e:
            violations.append(f"{py_file}: Parse error ({e})")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "evaluation" or alias.name.startswith("evaluation."):
                        violations.append(
                            f"{py_file.name}:{node.lineno} -> 'import {alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "evaluation" or (
                    node.module and node.module.startswith("evaluation.")
                ):
                    violations.append(
                        f"{py_file.name}:{node.lineno} -> 'from {node.module} import ...'"
                    )

    assert not violations, (
        "Strict Architectural Boundary Violated! 'src/' must NEVER import from 'evaluation/'.\n"
        "Comparison against EMSR927/CEMS must remain strictly isolated in evaluation/.\n"
        "Violations found:\n" + "\n".join(violations)
    )
