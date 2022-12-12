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
* Tested on Pop OS 20.04 LTS (you might find GUI elements being visually different on Windows or Mac OS)

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

Build to Windows Installation
-----------------------------

This application can be exported to a traditional executable program for various operating system. This can be helpful for distribution to individuals that are unfamiliar with Python and its environment setup.
To package this project into a portable executable program, we utilize **PyInstaller** to isolate the Python interpreter and the Python packages utilized in the Augmentation GUI.

We have created a simple specification file (:code:`app.sec`) that helps gather different packages properly. Run the following command to execute the spec file:
:code:`pyinstaller app.spec`

PyInstaller will output an isolated version of the Augmentation GUI into a folder called "dist". If you take a look into the folder, you will see all the familiar Python project files in addition to

Additionally, we use **Inno Setup 6** to take the PyInstaller output and convert it into a Windows-specific file, such as a Setup Wizard for installation. Inno Setup also wraps everything up into a single exe file that many Windows users are familiar with.