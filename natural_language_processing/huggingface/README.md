Please run the following command to make modules from this repository available in the project
```
export PYTHONPATH=/path/to/model_zoo
```

In order to use the script in the following directory for easy benchmarking of NLP models, you need to have 
OnSpecta's fork of Transformers repository.

Please run the following commands
```
# Install rust compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Make rust compiler available 
source $HOME/.cargo/env

git submodule update --init --recursive
cd huggingface/transformers
pip install -e .
```

Now you can run the benchmarking script. You can run it in the following manner
```
python3 huggingface/run.py -m bert-base-uncased -p fp32 -b 8 --sequence_length 8
```

Common issues
## Missing utils.misc
```
ModuleNotFoundError: No module named 'utils.misc'
```
make sure your PYTHONPATH is set

make sure you don't have "utils" package installed: 

`pip uninstall utils`

## libGLdispatch error
```
ImportError: /lib/aarch64-linux-gnu/libGLdispatch.so.0: cannot allocate memory in static TLS block
```
Solution

`export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1`
