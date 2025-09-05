"""FM2PROF package."""

__version__ = "2.4.0"

from fm2prof.fm2prof_runner import Project

# This is to tell ruff linter to respect the re-export
__all__ = ["Project"]
