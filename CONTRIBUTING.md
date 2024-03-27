### Each new model added should come with:
- ability to measure accuracy on a representable dataset containing real data (no accuracy measuring through feeding input tensors with noise and comparing to expected output tensor musters)
- ability to use AIO by the means of at least one of the Ampere optimized frameworks available
- unittest added to a respective file under tests/ directory testing model's functioning and accuracy

### Each merge to main should:
- ensure all .py files have the Ampere Computing ... copyright header in the top two lines
- ensure licensing compliance regarding all re-used / modified code, git submodules added, files linked, ... - see files in /licensing directory, you should also update LICENSE file with Copyright shoutout to the licensor
- pass CI test, including lint - max 120 chars per line (exception allowed in cases where wrapping will hurt readability rather than enhance it, put # noqa at the end of your line), adherence to most of the Python stylistic guidelines, no non-sense like unnecessary imports and variable initializations
- IMPROVE NOT DEGRADE

### Thanks for contributing and keeping up the good work!
