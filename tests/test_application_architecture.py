from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src" / "everypixel_cli"
APPLICATION = ROOT / "application"


def tree_for(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_roots(tree: ast.Module) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def called_functions(tree: ast.Module) -> list[ast.expr]:
    return [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_application_modules_do_not_depend_on_cli_or_rendering_frameworks():
    forbidden = {"typer", "click", "rich"}
    for path in APPLICATION.glob("*.py"):
        for node in ast.walk(tree_for(path)):
            if isinstance(node, ast.Import):
                assert not {name.name.split(".")[0] for name in node.names} & forbidden
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden


def test_application_modules_do_not_print_or_exit_processes():
    forbidden_calls = {"print", "exit", "quit"}
    for path in APPLICATION.glob("*.py"):
        for node in ast.walk(tree_for(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sys"
                    and node.func.attr == "exit"
                )


def test_cli_has_no_direct_http_transport_or_downloader_calls():
    tree = tree_for(ROOT / "cli.py")
    calls = called_functions(tree)
    assert not {
        call.attr
        for call in calls
        if isinstance(call, ast.Attribute) and call.attr in {"request", "get_status"}
    }
    assert not any(
        isinstance(call, ast.Attribute)
        and call.attr == "wait"
        and isinstance(call.value, ast.Attribute)
        and call.value.attr == "tasks"
        for call in calls
    )
    assert "download_task_result" not in {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert not any(
        isinstance(call, ast.Attribute)
        and call.attr == "client"
        and isinstance(call.value, ast.Name)
        and call.value.id == "runtime"
        for call in calls
    )


def test_cli_delegates_api_use_cases_to_application_services():
    tree = tree_for(ROOT / "cli.py")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    called_names = {
        call.id if isinstance(call, ast.Name) else call.attr
        for call in called_functions(tree)
        if isinstance(call, (ast.Name, ast.Attribute))
    }

    assert "OperationRequest" not in names
    assert "operations" not in attributes
    assert "submit_async" not in called_names
    assert not {
        name
        for name in called_names
        if name.startswith("build_") and name.endswith("_payload")
    }


def test_application_package_exports_only_high_level_api():
    tree = tree_for(APPLICATION / "__init__.py")
    exports = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )

    assert isinstance(exports, ast.List)
    assert {item.value for item in exports.elts if isinstance(item, ast.Constant)} == {
        "ApplicationServices",
        "ExecutionOptions",
        "OperationResult",
    }


def test_mcp_server_uses_application_services_instead_of_transport_calls():
    tree = tree_for(ROOT / "mcp_server.py")
    calls = called_functions(tree)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "typer" not in imported_roots(tree)
    assert not {
        call.attr
        for call in calls
        if isinstance(call, ast.Attribute) and call.attr in {"request", "get_status"}
    }
    assert "OperationRequest" not in names
    assert not any(
        keyword.arg == "endpoint"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
        and keyword.value.value.startswith("/v1/")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )
