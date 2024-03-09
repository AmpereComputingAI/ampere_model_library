# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import subprocess
import sys
import json
import pathlib
import argparse
from datetime import date
from urlextract import URLExtract

DOMAINS_TO_CHECK = ["ampereaimodelzoo", "ampereaidevelop", "ampereaidevelopus", "dropbox"]
HEADER_IGNORE = [".git", "LICENSE"]
HEADER_IGNORE_EXTENSIONS = ["md", "sample", "xml", "json", "html", "txt", "ipynb", "sh"]


def get_issue_printer():
    issue_idx = 0

    def print_issue(text):
        nonlocal issue_idx
        print(f"[Issue #{issue_idx}] {text}")
        issue_idx += 1
        return True

    return print_issue


def get_header(year):
    return ["# SPDX-License-Identifier: Apache-2.0", f"# Copyright (c) {year}, Ampere Computing LLC"]


def check_headers():
    aml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    print_issue = get_issue_printer()
    failure = False

    with open(os.path.join(aml_dir, ".gitmodules"), "r") as f:
        submodules = [
            os.path.join(aml_dir, submodule.replace("\tpath = ", ""))
            for submodule in f.read().splitlines() if "path" in submodule
        ]
    paths_to_ignore = submodules + [os.path.join(aml_dir, path) for path in HEADER_IGNORE]

    all_paths = list(pathlib.Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")).rglob("*"))
    for path in all_paths:
        do_continue = False
        if path.is_dir() or str(path).split(".")[-1] in HEADER_IGNORE_EXTENSIONS:
            continue
        for path_to_ignore in paths_to_ignore:
            if path_to_ignore in str(path):
                do_continue = True
                break
        if do_continue:
            continue

        with open(path, "r") as f:
            try:
                lines = [f.readline().strip() for _ in range(2)]
            except UnicodeDecodeError:
                continue
            year_of_last_mod = int(
                subprocess.check_output(f"git log -1 --format=%cd --date=format:%Y -- {path}".split()).decode().strip())
            target_lines = get_header(year_of_last_mod)
            if lines != target_lines:
                failure = print_issue(
                    f"Ampere's copyright header missing in file {str(path).replace(aml_dir, '')}")

    if failure:
        header = "\n".join(get_header(date.today().year))
        print(f"\nFollowing copyright header should be placed in first two lines of each file:\n"
              f"\033[91m{header}\033[0m")
        sys.exit(1)


def check_links():
    aml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    approved_urls_file = "licensing/approved_urls.json"
    with open(os.path.join(aml_dir, approved_urls_file), "r") as f:
        approved_urls = json.load(f)["approved"]
    print_issue = get_issue_printer()
    failure = False

    all_urls = {}
    extractor = URLExtract()
    all_paths = list(pathlib.Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")).rglob("*"))
    for path in all_paths:
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
                        files[url].add(str(path).replace(aml_dir, ""))
                    else:
                        files[url] = {str(path).replace(aml_dir, "")}

    for url in missing_urls:
        failure = print_issue(
            f"\033[91m{url}\033[0m not listed in {approved_urls_file}, found in files: [{', '.join(files[url])}]")

    for url in missing_urls:
        print(f"\"{url}\",")

    if failure:
        sys.exit(1)


def check_submodules():
    aml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    approved_submodules_file = "licensing/approved_submodules.json"
    with open(os.path.join(aml_dir, approved_submodules_file), "r") as f:
        approved_submodules = json.load(f)["approved"]
    print_issue = get_issue_printer()
    failure = False

    with open(os.path.join(aml_dir, ".gitmodules"), "r") as f:
        submodules = [
            submodule.replace(
                "\tpath = ", "") for submodule in f.read().splitlines() if "path" in submodule
        ]

    for submodule in submodules:
        if submodule not in approved_submodules.keys():
            failure = print_issue(f"Submodule \033[91m{submodule}\033[0m not listed in {approved_submodules_file}")
        elif approved_submodules[submodule] is not None:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", submodule, "LICENSE"), "r") as f:
                for line in f.read().splitlines():
                    if approved_submodules[submodule] in line:
                        break
                else:
                    failure = print_issue(f"'{approved_submodules[submodule]}' not found in {submodule}/LICENSE file")

    copyright_mentions = {i: False for i, j in approved_submodules.items() if j is not None}
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "LICENSE"), "r") as f:
        for line in f.read().splitlines():
            if "Copyright (c)" in line and "Ampere Computing LLC" in line:
                current_year = date.today().year
                year_license = int(line.split()[2].replace(",", ""))
                if year_license > current_year:
                    failure = print_issue(f"Come back from the future pal! Set year for Ampere LLC in Copyright "
                                          f"section of LICENSE file to {current_year}.")
                elif year_license < current_year:
                    failure = print_issue(f"Happy New Year! Please set year for Ampere LLC in Copyright section of "
                                          f"LICENSE file to {current_year}.")
            else:
                for submodule, copyright_header in approved_submodules.items():
                    if copyright_header is None:
                        continue
                    if copyright_header in line:
                        copyright_mentions[submodule] = True
    for submodule, success in copyright_mentions.items():
        if not success:
            failure = print_issue(f"'{approved_submodules[submodule]}' not listed in AML's LICENSE file")

    if failure:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Intellectual property tester")
    parser.add_argument("--check", choices=["links", "modules", "headers"], required=True)
    args = parser.parse_args()
    if args.check == "links":
        check_links()
    elif args.check == "modules":
        check_submodules()
    elif args.check == "headers":
        check_headers()
    else:
        assert False
