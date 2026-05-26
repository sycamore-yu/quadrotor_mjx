"""Headless-safe checks for the mjviser play CLI."""

from __future__ import annotations

from scripts.visualize_mjviser import main


def test_visualize_check_scene_does_not_import_viewer_server(capsys):
    """Scene checks must work on headless servers without starting viser."""

    assert main(["--env", "hover_obstacle", "--mode", "scene", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "dry-run scene check passed" in output.lower()
    assert "hover_obstacle" in output
