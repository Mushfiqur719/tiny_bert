"""src/utils/test_guard.py — Guard against test set data leakage.

Ensures test labels and test splits cannot be accessed during model training
or hyperparameter tuning. Only scripts explicitly authorizing test evaluation
(such as scripts/final_test.py) may set TBB_ALLOW_TEST=1.
"""

from __future__ import annotations

import os


class TestSetLocked(RuntimeError):
    """Raised when code attempts to access the test set without authorization."""
    __test__ = False


def assert_test_access() -> None:
    """Verify that test set access is explicitly authorized via environment variable."""
    if os.environ.get("TBB_ALLOW_TEST") != "1":
        raise TestSetLocked(
            "Test labels are locked. Only scripts/final_test.py may set TBB_ALLOW_TEST=1."
        )
