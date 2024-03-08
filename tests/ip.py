import os
import sys
from datetime import date

# before adding make sure that the licensing permits legal use in AML
APPROVED_SUBMODULES = {
    "utils/cv/nnUNet": "Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), "
                       "Heidelberg, Germany]",
    "utils/recommendation/dlrm": "Copyright (c) Facebook, Inc. and its affiliates.",
    "computer_vision/semantic_segmentation/segment_anything/segment_anything": None,
    "natural_language_processing/text_generation/nanogpt/nanoGPT": "Copyright (c) 2022 Andrej Karpathy",
    "utils/recommendation/DeepCTR": "Copyright 2017-present Weichen Shen",
    "speech_recognition/whisper/whisper": "Copyright (c) 2022 OpenAI",
    "text_to_image/stable_diffusion/stablediffusion": "Copyright (c) 2022 Stability AI",
    "speech_recognition/whisper/openai_whisper": "Copyright (c) 2022 OpenAI"
}


def get_issue_printer():
    issue_idx = 0

    def print_issue(text):
        nonlocal issue_idx
        print(f"[Issue #{issue_idx}] {text}")
        issue_idx += 1
        return True

    return print_issue


def run_legal_audit():
    print_issue = get_issue_printer()
    failure = False

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".gitmodules"), "r") as f:
        submodules = [
            submodule.replace(
                "\tpath = ", "") for submodule in f.read().splitlines() if "path" in submodule
        ]

    for submodule in submodules:
        if submodule not in APPROVED_SUBMODULES.keys():
            failure = print_issue(f"Submodule {submodule} not in APPROVED_SUBMODULES")
        elif APPROVED_SUBMODULES[submodule] is not None:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", submodule, "LICENSE"), "r") as f:
                for line in f.read().splitlines():
                    if APPROVED_SUBMODULES[submodule] in line:
                        break
                else:
                    failure = print_issue(f"'{APPROVED_SUBMODULES[submodule]}' not found in {submodule}/LICENSE file")

    copyright_mentions = {i: False for i, j in APPROVED_SUBMODULES.items() if j is not None}
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
                for submodule, copyright_header in APPROVED_SUBMODULES.items():
                    if copyright_header is None:
                        continue
                    if copyright_header in line:
                        copyright_mentions[submodule] = True
    for submodule, success in copyright_mentions.items():
        if not success:
            failure = print_issue(f"'{APPROVED_SUBMODULES[submodule]}' not listed in AML's LICENSE file")

    if failure:
        sys.exit(1)


if __name__ == "__main__":
    run_legal_audit()
