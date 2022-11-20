Installation Process
==========================

Source Installation
-----------------

To recreate the development environment to run and edit this project, it is highly recommended you installed Anaconda for package and Python version management. Creating an venv will also work, but pip package versions might not have correct conflict resolutions.

**Optimal Environment:**

* Python 3.7
* CUDA-enabled GPU (optional)
* Qt5 Packages (can be installed through `conda`)
* PyTorch >= 1.7 (and associating Torchvision version)
* Tested on Pop OS 20.04 LTS (you might find GUI elements misplaced on Windows and Mac OS)

**Anaconda Steps:**

1. Start an Anaconda shell
2. Create an environment: :code:`conda create --name noisey python=3.7`
3. Installing FFMPEG and libx264: :code:`conda install x264 ffmpeg=4.0.2 -c conda-forge`
4. Installing torch:
    - (CUDA): :code:`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` (adjust CUDA versions to your system's driver version)
    - (CPU-only): :code:`conda install pytorch torchvision torchaudio cpuonly -c pytorch`
5. Install required packages: :code:`pip install -r requirements.txt`

**virtualenv Steps:**

1. Create an environment: :code:`python3 -m venv [NAME_OF_ENVIRONMENT]`
2. Activate environment (OS-dependent; `Link more info here - scroll down to "Activate [...]" section <https://www.infoworld.com/article/3239675/virtualenv-and-venv-python-virtual-environments-explained.html>`_)
3. Install packages via pip: :code:`pip install -r requirements.txt`
    - NOTE: If pip complains about versions not existing and displays a version warning message, please update pip in your venv