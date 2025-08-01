import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ssip",
    version="0.1",
    author="Shahed & Bilal",
    author_email="{ahmed348, ahmedb}@purdue.edu",
    description="Synthesizing speech from Ecog as inverse problem, using diffusion model as a prior.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shahed517/SSIP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'matplotlib'],
)