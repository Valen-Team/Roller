"""Extract a sandbox transfer without following links or restoring ownership."""

from __future__ import annotations

import io
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


def unpack_archive(archive: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError("archive destination is a symbolic link")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    directories = []
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"archive entry escapes the transfer root: {member.name}"
                )
            if not relative.parts and member.isdir():
                continue
            target = root.joinpath(*relative.parts)
            paths = [target, *target.parents]
            for path in paths[: paths.index(root)]:
                if path.is_symlink():
                    raise ValueError(
                        f"archive entry crosses a symbolic link: {member.name}"
                    )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directories.append((target, member.mode & 0o777))
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                assert isinstance(stream, io.BufferedIOBase)
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(member.mode & 0o777)
            else:
                raise ValueError(
                    f"archive entry is not a file or directory: {member.name}"
                )
    for path, mode in reversed(directories):
        path.chmod(mode)


if __name__ == "__main__":
    unpack_archive(Path(sys.argv[1]), Path(sys.argv[2]))
