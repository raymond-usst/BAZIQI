"""Path safety helpers: ensure file I/O stays under intended directories."""

import os


def safe_under(base_dir: str, path: str) -> bool:
    """
    Return True if the resolved path is under base_dir (no path traversal escape).
    Uses absolute paths so that .. and absolute paths are correctly rejected.
    """
    if not base_dir:
        return False
    base_abs = os.path.abspath(os.path.normpath(base_dir))
    path_abs = os.path.abspath(os.path.normpath(path))
    # path must be under base (same dir or a subdir)
    try:
        common = os.path.commonpath([base_abs, path_abs])
    except ValueError:
        return False
    return common == base_abs and (path_abs == base_abs or path_abs.startswith(base_abs + os.sep))


def resolve_under(base_dir: str, path: str) -> str:
    """
    Resolve path and return it if it is under base_dir; else raise ValueError.
    """
    if not safe_under(base_dir, path):
        raise ValueError("Path escapes base directory")
    return os.path.abspath(os.path.normpath(path))
