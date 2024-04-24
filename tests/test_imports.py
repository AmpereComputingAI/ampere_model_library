# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import unittest
import pathlib
from tests.test_ip import IssueTracker

EXCEPTIONS_CHECKER = ["/demos/privacy_preservation/run.py"]
EXCEPTIONS_IMPORTS = ["/demos/privacy_preservation/run.py"]


class TestImports(unittest.TestCase):
    def setUp(self):
        self.filedir = os.path.dirname(os.path.realpath(__file__))
        self.aml_dir = os.path.join(self.filedir, "..")
        self.is_func_declaration = lambda x: any(
            [token in x for token in ["def ", "if __name__ == \"__main__\":", "if __name__ == '__main__'':"]])
        with open(os.path.join(self.filedir, "env_variables_checker.txt"), "r") as f:
            self.codeblock = f.readlines()

    def test_for_env_variables_setup_checker(self):
        issue_tracker = IssueTracker()
        for path in list(pathlib.Path(self.aml_dir).rglob("*")):
            if (not path.is_file() or
                    any([token not in path.name for token in [".py", "run"]]) or
                    str(path).replace(self.aml_dir, '') in EXCEPTIONS_CHECKER):
                continue
            block_idx = 0
            with open(path, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line == self.codeblock[block_idx]:
                        block_idx += 1
                        if block_idx == len(self.codeblock):
                            break
                    elif block_idx > 0:
                        if os.environ.get("FIX") == "1":
                            lines[i] = self.codeblock[block_idx]
                            block_idx += 1
                            if block_idx == len(self.codeblock):
                                break
                        else:
                            issue_tracker.register(f"Env variables checker codeblock corrupted in file \033[91m"
                                                   f"{str(path).replace(self.aml_dir, '')}\033[0m")
                            break
                    if self.is_func_declaration(line):
                        issue_tracker.register(f"Env variables checker codeblock not found in file \033[91m"
                                               f"{str(path).replace(self.aml_dir, '')}\033[0m")
                        break
            if os.environ.get("FIX") == "1":
                with open(path, "w") as f:
                    f.writelines(lines)
        failure, issues = issue_tracker.is_failure()
        self.assertFalse(
            failure,
            issues + f"\nAll runner files should be prepended with following codeblock:\n\n"
                     f"\033[91m{''.join(self.codeblock)}\033[0m")


    def test_no_imports_top(self):
        issue_tracker = IssueTracker()
        for path in list(pathlib.Path(self.aml_dir).rglob("*")):
            if (not path.is_file() or
                    any([token not in path.name for token in [".py", "run"]]) or
                    str(path).replace(self.aml_dir, '') in EXCEPTIONS_IMPORTS):
                continue
            inside_func = False
            block_idx = 0
            with open(path, "r") as f:
                for line in f.readlines():
                    if line == self.codeblock[block_idx] and block_idx >= 0:
                        inside_func = True
                        block_idx += 1
                        if block_idx == len(self.codeblock):
                            block_idx = -1
                            inside_func = False
                    if "import" in line and not inside_func:
                        issue_tracker.register(f"Import call(s) found in file \033[91m"
                                               f"{str(path).replace(self.aml_dir, '')}\033[0m")
                        break
                    if self.is_func_declaration(line):
                        inside_func = True
                    elif inside_func and all([token != line[0] for token in [" ", "\n"]]):
                        inside_func = False
        failure, issues = issue_tracker.is_failure()
        self.assertFalse(failure, issues + "\nImports should not be done outside of functions in runner files.")
