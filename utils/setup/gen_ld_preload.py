# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import re
import subprocess
from pathlib import Path


# generate the list of preload files
script_dir = os.path.dirname(os.path.realpath(__file__))
ld_preload = list()

for path in Path("/").rglob("libgomp*"):
    if ".so" in path.name:
        ld_preload.append(str(path))

for path in Path("/").rglob("libGLdispatch.so.0"):
    ld_preload.append(str(path))

# test the preload for errors
os.environ["LD_PRELOAD"] = ":".join(ld_preload)
test_cmd = ["python3", "-c", "'print(\"AML\")'"]

process = subprocess.Popen(test_cmd, stderr=subprocess.PIPE)
_, errors = process.communicate()
errors = errors.decode("utf-8")

start_pattern = "object '"
start_indices = [match.start() + len(start_pattern) for match in re.finditer(start_pattern, errors)]

end_pattern = "' from LD_PRELOAD cannot be preloaded"
end_indices = [match.start() for match in re.finditer(end_pattern, errors)]

for start, end in zip(start_indices, end_indices):
    ld_preload.remove(errors[start:end])

f = open(os.path.join(script_dir, ".ld_preload"), "w")
f.write(":".join(ld_preload))
f.close()
