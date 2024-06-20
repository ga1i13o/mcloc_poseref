import gdown
import os
import shutil
import urllib.request


cambridge_scenes = {
    'OldHospital': 'https://www.repository.cam.ac.uk/bitstreams/ae577bfb-bdce-488c-8ce6-3765eabe420e/download',
    'KingsCollege': 'https://www.repository.cam.ac.uk/bitstreams/1cd2b04b-ada9-4841-8023-8207f1f3519b/download',
    'StMarysChurch': 'https://www.repository.cam.ac.uk/bitstreams/2559ba20-c4d1-4295-b77f-183f580dbc56/download',
    'ShopFacade': 'https://www.repository.cam.ac.uk/bitstreams/4e5c67dd-9497-4a1d-add4-fd0e00bcb8cb/download'
}
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

## Download splatting models for Cambridge
print('Downloading Gaussian Splatting models for cambrdige...')
url = 'https://drive.google.com/uc?id=1iCyizI0jZwZ7mdXG9wGGh2fuwdsbQVkL'
output = 'g_down.zip'
gdown.download(url, output, quiet=False)

unzip_cmd = f'unzip {output}'
os.system(unzip_cmd)
os.rename('out_gp', 'cambridge_splats')
shutil.move('cambridge_splats', DATA_DIR)
os.remove(output) # remove zip file

# Download undistorted colmap models for Cambridge and 7scenes
print('Downloading colmap models for Cambridge and 7 scenes...')
url = 'https://drive.google.com/uc?id=1BPUU7Z_Xc4SIQJwfIiIHwUA9ud9c9tCU'
output = 'all_colmaps.zip'
gdown.download(url, output, quiet=False)
unzip_cmd = f'unzip {output}'
os.system(unzip_cmd)
shutil.move('all_colmaps', DATA_DIR)
os.remove(output) # remove zip file

## Download cambridge scenes
wget_cmd = 'wget {url} -O {out_name}'
for cs, cs_url in cambridge_scenes.items():
    print(f'Downloading dataset for {cs}...')
    # urllib.request.urlretrieve(cs_url, f'{cs}.zip')
    os.system(wget_cmd.format(url=cs_url, out_name=f'{cs}.zip'))
    unzip_cmd = f'unzip {cs}.zip'
    os.system(unzip_cmd)
    shutil.move(cs, DATA_DIR)
    os.remove(f'{cs}.zip') # remove zip file
