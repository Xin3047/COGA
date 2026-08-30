from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from coga.rst import _validate_tar_members


def test_tar_validation_rejects_parent_traversal(tmp_path: Path) -> None:
    member = tarfile.TarInfo("../escape.txt")
    with pytest.raises(ValueError, match="unsafe TAR member"):
        _validate_tar_members([member], tmp_path)


def test_tar_validation_accepts_regular_relative_file(tmp_path: Path) -> None:
    member = tarfile.TarInfo("tasks/example/instruction.md")
    _validate_tar_members([member], tmp_path)
