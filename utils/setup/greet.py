import ssl
from urllib.request import urlopen


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


with open("/.release", "r") as release_file:
    release_meta = release_file.read()

this_release = release_meta.split("release")[1].replace(" ", "").replace("\n", "")
latest_release = this_release

latest_release_url = None
if "PYTORCH" in release_meta:
    latest_release_url = "https://raw.githubusercontent.com/AmpereComputingAI/releases_meta/main/pytorch.txt"
elif "TF" in release_meta:
    latest_release_url = "https://raw.githubusercontent.com/AmpereComputingAI/releases_meta/main/tensorflow.txt"
elif "ONNXRT" in release_meta:
    latest_release_url = "https://raw.githubusercontent.com/AmpereComputingAI/releases_meta/main/onnxruntime.txt"

if latest_release_url is not None:
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        latest_release = urlopen(latest_release_url).readline().decode('ascii').replace(" ", "").replace("\n", "")
    except:
        pass

try:
    ssl._create_default_https_context = ssl._create_unverified_context
    logo = urlopen("https://raw.githubusercontent.com/AmpereComputingAI/ampere_model_library/main/utils/setup/aio_logo.txt").read().decode('ascii')
except:
    logo = None

if logo is not None:
    CRED = '\033[91m'
    CEND = '\033[0m'

    colorized = "\n"
    rows = 15
    cols = 71

    for n in range(rows):
        colorized += CRED
        for i in range(cols):
            colorized += logo[n*cols+i]
            if i > 46:
                colorized += CEND

    print(colorized)

if versiontuple(this_release) >= versiontuple(latest_release):
    print("\nThank you for choosing AIO!")
else:
    print(f"\n{CRED}New version of AIO available!\nYou are using release {this_release}, while latest release is {latest_release}.{CEND}")
print("Please visit us at https://solutions.amperecomputing.com/solutions/ampere-ai")

print(f"\nQuickstart READ-ME: https://github.com/AmpereComputingAI/ampere_model_library?tab=readme-ov-file#examples")
