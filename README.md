# MOOC_pymat
This repository contains the [Manim](https://www.manim.community/) code used for creating the animations for the MOOC: **Getting started hands on learning: Python and Matlab**. The animations are converted to a Power-Point presentation using [Manim-Slides](https://www.manim.community/plugin/manim-slides/).
Additionally, some of the supplementary material of the course is present.

## Installation
First, clone the repo locally:
```shell
git clone https://github.com/ReFraschiniAndrea/MOOC_pymat.git
```

It issuggested to create a python virtual environment before proceeding. Then, install the necessary dependencies.
```shell
pip install manim
pip install manim-slides
pip install scikit-image
```

Afterward, a few custom [Pygments](https://pygments.org/) styles and lexers are needed to display the code in the animations. To install them as a plugin, run the following from the root of the repo:
```shell
pip install --no-build-isolation -e MOOC_pygments
```

### Latex distribution
A Latex distribution is needed for compiling the latex formulas. We suggest to use [MikTex](https://miktex.org/).
After installing it, make sure to open the Miktek console, check for any available updates and install them; this will make sure that the necessary packages are downloaded.

> [!NOTE]
> If during rendering, when attempting to compile Latex, you get the following gerror:\
> `'latex' is not recognized as an internal or external command, operable program or batch file`\
> This probably means that the latex executable is not on the current `PATH` environment variable. Try to add it explicitlwith this following line in the `mooc_utils\__init__.py`:
> ```python
> os.environ["PATH"] = r"C:\Path\to\Latex\Folder;" + os.environ["PATH"]
> ```

### Manim-slides in 4:3
At the moment, `manim-slides` support for Power Point presentations is not complete, and only slides with an aspect ratio of 16:9 can be created. Since the animations are instead intended for 4:3 slides, some hacking is required.

Navigate to the source code of manim-slides in your virtual environment: from the root of the environment, it should be in `Lib\site-packages\manim-slides`.
Then, modify the `convert.py` file found there by changing the values 1280 and 720 found in lines 733 and 737 to 1440 and 1080 respectively.

## Usage
To render the animations of a particular week, position yourself in the main folder and run `manim-slides render` with the path to the animations you want to render. For example:
```bash
manim-slides render .\WEEK_1.\W2Theory_slides.py
```
will render all the partial animation videos of the theory section of the second week. To then create the Power Point presentation, run `manim-slides convert`:
```bash
manim-slides convert --to=pptx W2Theory_slides week2_theory_slides.pptx
```
