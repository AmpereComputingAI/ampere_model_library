import gzip
import json

with gzip.open("../logs/plugins/profile/2021_06_15_15_34_43/49524a0fc1dd.trace.json.gz", "r") as f:
    data = f.read()
    j = json.loads(data.decode('utf-8'))
    app_json = json.dumps(j)

test
f = open("demofile2.txt", "w")
f.write(app_json)
f.close()
