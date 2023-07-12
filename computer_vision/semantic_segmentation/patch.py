import os
import inspect
import shutil
import sys
try:
    import segment_anything
except:
    print('You need to install segment_anything from facebook first using')
    print('pip install git+https://github.com/facebookresearch/segment-anything.git')
    quit()

current_folder = os.path.dirname(sys.argv[0])

librarypath = os.path.dirname(inspect.getsourcefile(segment_anything))
shutil.copy(os.path.join(current_folder, 'patch/mask_decoder.py'), 
            os.path.join(librarypath, 'modeling/mask_decoder.py'))
shutil.copy(os.path.join(current_folder,'patch/prompt_encoder.py'), 
            os.path.join(librarypath, 'modeling/prompt_encoder.py'))
print('segment anything is patched properly.')
