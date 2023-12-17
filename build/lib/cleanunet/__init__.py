import json
import torch
from .network import CleanUNet

import os
import requests
from tqdm import tqdm

def download_file(url, cache_dir='cache'):
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Extract filename from URL
    filename = url.split('/')[-1]
    file_path = os.path.join(cache_dir, filename)

    # Check if file is already in cache
    if os.path.exists(file_path):
        print(f"Loading from cache: {file_path}")
        return file_path

    # Download and save the file with a progress bar
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as file, tqdm(
        desc=filename, total=total_size, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print(f"Saved to cache: {file_path}")
    return file_path

# Example usage
# downloaded_file = download_file('https://example.com/path/to/file.txt')

urls = {
    'full':{
        'config': "https://github.com/sawradip/CleanUNet-pip/raw/main/configs/DNS-large-full.json",
        'weights': "https://github.com/sawradip/CleanUNet-pip/raw/main/exp/DNS-large-full/checkpoint/pretrained.pkl"
    },
    'high':{
        'config': "https://github.com/sawradip/CleanUNet-pip/raw/main/configs/DNS-large-high.json",
        'weights': "https://github.com/sawradip/CleanUNet-pip/raw/main/exp/DNS-large-high/checkpoint/pretrained.pkl"
    }
}

def load_config(config_path):
    with open(config_path) as f:
        data = f.read()
        config = json.loads(data)
    gen_config              = config["gen_config"]
    network_config          = config["network_config"]      # to define wavenet
    train_config            = config["train_config"]        # train config
    trainset_config         = config["trainset_config"]     # to read trainset configurations
    return gen_config, network_config, train_config, trainset_config

@classmethod
def from_pretarined(cls, varient = 'full', device = 'cpu'):
    device = torch.device(device)
    config_path = download_file(urls[varient]['config'])
    weights_path = download_file(urls[varient]['weights'])
    
    _, network_cfg, _, _ = load_config(config_path)
    
    net = CleanUNet(**network_cfg).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    return net

CleanUNet.from_pretrained = from_pretarined
    
    