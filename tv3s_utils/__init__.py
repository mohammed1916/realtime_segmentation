"""tv3s_utils package init.

Avoid importing submodules with side-effects at package import time. Tests
and tooling should import submodules explicitly (for example,
`import tv3s_utils.utils.datasets.transforms as tr`) to prevent
unexpected registration of models or other globals.
"""

__all__ = []