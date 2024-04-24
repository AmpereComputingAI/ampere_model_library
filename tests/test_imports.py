# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import unittest
import pathlib
from tests.test_ip import IssueTracker

EXCEPTIONS = ["/demos/privacy_preservation/run.py"]


class TestImports(unittest.TestCase):
    def test_no_imports_top(self):
        aml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
        issue_tracker = IssueTracker()
        for path in list(pathlib.Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")).rglob("*")):
            if (not path.is_file() or
                    any([token not in path.name for token in [".py", "run"]]) or
                    str(path).replace(aml_dir, '') in EXCEPTIONS):
                continue
            inside_func = False
            with open(path, "r") as f:
                for line in f.readlines():
                    if "import" in line and not inside_func:
                        issue_tracker.register(f"Import call(s) found in file \033[91m"
                                               f"{str(path).replace(aml_dir, '')}\033[0m")
                        break
                    if any([token in line for token in
                            ["def ", "if __name__ == \"__main__\":", "if __name__ == '__main__'':"]]):
                        inside_func = True
                    elif inside_func and all([token != line[0] for token in [" ", "\n"]]):
                        inside_func = False
        failure, issues = issue_tracker.is_failure()
        self.assertFalse(failure, issues + "\nImports should not be done outside of functions in runner files.")
