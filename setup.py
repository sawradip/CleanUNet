from setuptools import setup, find_packages

VERSION = '0.0.2' 
DESCRIPTION = 'CleanUNet - speech Denoiser'
LONG_DESCRIPTION = 'pip installable version of - Official PyTorch Implementation of CleanUNet (ICASSP 2022) '

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="cleanunet", 
        version=VERSION,
        author="Sawradip Saha",
        author_email="<sawradip0@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'pillow',
            'torchaudio',
            'inflect',
            'scipy',
            'tqdm',
            'pesq',
            'pystoi'
            ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'speech-denoiser'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)