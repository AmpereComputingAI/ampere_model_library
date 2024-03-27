# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import subprocess
import unittest
import json
import pathlib
from datetime import date
from urlextract import URLExtract

DOMAINS_TO_CHECK = ["ampereaimodelzoo", "ampereaidevelop", "ampereaidevelopus", "dropbox"]
HEADER_IGNORE = [".git", "LICENSE"]
HEADER_EXTENSIONS = ["py"]


# HEADER_IGNORE_EXTENSIONS = ["md", "sample", "xml", "json", "html", "txt", "ipynb", "sh"]


class IssueTracker:
    def __init__(self):
        self._issues = []

    def register(self, text):
        self._issues.append(f"[Issue #{len(self._issues)}] {text}")
        return True

    def is_failure(self):
        if len(self._issues) > 0:
            return True, "\n" + "\n".join(self._issues) + "\n"
        else:
            return False, ""


def get_header(year):
    return ["# SPDX-License-Identifier: Apache-2.0", f"# Copyright (c) {year}, Ampere Computing LLC"]


class TestIntellectualPropertyCompliance(unittest.TestCase):
    def setUp(self):
        subprocess.run("git submodule update --init --recursive".split(), check=True)
        self.all_paths = list(pathlib.Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")).rglob("*"))
        self.aml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

    def test_copyright_headers(self):
        issue_tracker = IssueTracker()

        with open(os.path.join(self.aml_dir, ".gitmodules"), "r") as f:
            submodules = [os.path.join(self.aml_dir, submodule.replace("\tpath = ", ""))
                          for submodule in f.read().splitlines() if "path" in submodule]
        paths_to_ignore = submodules + [os.path.join(self.aml_dir, path) for path in HEADER_IGNORE]

        for path in self.all_paths:
            do_continue = False
            if path.is_dir() or str(path).split(".")[-1] not in HEADER_EXTENSIONS:
                continue
            for path_to_ignore in paths_to_ignore:
                if path_to_ignore in str(path):
                    do_continue = True
                    break
            if do_continue:
                continue

            with open(path, "r") as f:
                try:
                    lines = ["# " + f.readline().split("#")[1].strip() for _ in range(2)]
                except UnicodeDecodeError:
                    continue
                except IndexError:
                    if len(f.readlines()) == 0:
                        continue
                    lines = []
                git_cmd = f"git log -1 --format=%cd --date=format:%Y -- {path}"
                try:
                    target_lines = get_header(int(subprocess.check_output(git_cmd.split()).decode().strip()))
                except ValueError:
                    continue
                if lines != target_lines:
                    issue_tracker.register(f"Header missing or out-dated in file \033[91m"
                                           f"{str(path).replace(self.aml_dir, '')}\033[0m")

        failure, issues = issue_tracker.is_failure()
        header = "\n".join(get_header(date.today().year))
        self.assertFalse(
            failure, issues + f"\nFollowing copyright header should be placed in first two lines of each listed file:"
                              f"\n\033[91m{header}\033[0m")

    def test_links(self):
        issue_tracker = IssueTracker()

        approved_urls_file = "licensing/approved_urls.json"
        with open(os.path.join(self.aml_dir, approved_urls_file), "r") as f:
            approved_urls = json.load(f)["approved"]

        all_urls = {}
        extractor = URLExtract()
        for path in self.all_paths:
            if path.is_dir():
                continue
            with open(path, "r") as f:
                try:
                    urls = extractor.find_urls(f.read())
                except UnicodeDecodeError:
                    continue
                if len(urls) > 0:
                    all_urls[path] = urls

        files = {}
        missing_urls = set()
        for path, urls in all_urls.items():
            for url in urls:
                for restricted_domain in DOMAINS_TO_CHECK:
                    if restricted_domain in url and url not in approved_urls:
                        missing_urls.add(url)
                        if url in files.keys():
                            files[url].add(str(path).replace(self.aml_dir, ""))
                        else:
                            files[url] = {str(path).replace(self.aml_dir, "")}

        for url in missing_urls:
            issue_tracker.register(
                f"\033[91m{url}\033[0m not listed in {approved_urls_file}, found in files: [{', '.join(files[url])}]")

        self.assertFalse(*issue_tracker.is_failure())

    def test_submodules(self):
        issue_tracker = IssueTracker()

        approved_submodules_file = "licensing/approved_submodules.json"
        with open(os.path.join(self.aml_dir, approved_submodules_file), "r") as f:
            approved_submodules = json.load(f)["approved"]

        with open(os.path.join(self.aml_dir, ".gitmodules"), "r") as f:
            submodules = [
                submodule.replace(
                    "\tpath = ", "") for submodule in f.read().splitlines() if "path" in submodule
            ]

        for submodule in submodules:
            if submodule not in approved_submodules.keys():
                issue_tracker.register(
                    f"Submodule \033[91m{submodule}\033[0m not listed in {approved_submodules_file}")
            elif approved_submodules[submodule] is not None:
                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", submodule, "LICENSE"),
                          "r") as f:
                    for line in f.read().splitlines():
                        if approved_submodules[submodule] in line:
                            break
                    else:
                        issue_tracker.register(
                            f"'{approved_submodules[submodule]}' not found in {submodule}/LICENSE file")

        copyright_mentions = {i: False for i, j in approved_submodules.items() if j is not None}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "LICENSE"), "r") as f:
            for line in f.read().splitlines():
                if "Copyright (c)" in line and "Ampere Computing LLC" in line:
                    current_year = date.today().year
                    year_license = int(line.split()[2].replace(",", ""))
                    if year_license > current_year:
                        issue_tracker.register(
                            f"Come back from the future pal! Set year for Ampere LLC in Copyright section of LICENSE "
                            f"file to {current_year}.")
                    elif year_license < current_year:
                        issue_tracker.register(
                            f"Happy New Year! Please set year for Ampere LLC in Copyright section of LICENSE file to "
                            f"{current_year}.")
                else:
                    for submodule, copyright_header in approved_submodules.items():
                        if copyright_header is None:
                            continue
                        if copyright_header in line:
                            copyright_mentions[submodule] = True
        for submodule, success in copyright_mentions.items():
            if not success:
                issue_tracker.register(f"'{approved_submodules[submodule]}' not listed in AML's LICENSE file")

        self.assertFalse(*issue_tracker.is_failure())


if __name__ == "__main__":
    unittest.main()
