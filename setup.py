from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="canyonbpy",
    version="0.1.0",
    description="Python implementation of CANYON-B for oceanographic parameter predictions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Raphaël Bajon",
    author_email="raphael.bajon@ifremer.fr",
    url="https://github.com/RaphaelBajon/canyonbpy",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'canyonbpy': ['data/weights/*.txt']
    },
    package_dir={"canyonbpy": "canyonbpy"},
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Oceanography',
    ],
    keywords='oceanography, CANYON-B, neural networks, carbon, nutrients',
    project_urls={
        'Bug Reports': 'https://github.com/RaphaelBajon/canyonbpy/issues',
        'Source': 'https://github.com/RaphaelBajon/canyonbpy',
    },
)