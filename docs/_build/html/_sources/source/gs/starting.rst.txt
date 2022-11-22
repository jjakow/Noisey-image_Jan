Introduction
==========================

Application Purpose
-----------------

This program is a flexible and extensive tool for researchers to conduct sensitivity and benchmark analysis of neural networks on augmented data.
The main features of the program is as follows:

* Ability to evaluate different neural networks for object detection
* Apply wide range of image-based augmentations to entire datasets automatically and procedurally
* Automate sensitivity analysis experiments

.. image:: images/mainWindow.png
  :width: 750
  :alt: Main window of Noisey GUI

How It Works
-----------------

The application has a basic workflow:

1. Upload image(s) or dataset YAML files
2. Setup experiment with desired image augmentations and parameter ranges
3. Preview augmentations on a given image and preview detection 
4. Run user-defined experiment parameters and generate detections and graphing results

+++++++++++++++
Augmentation Pipeline
+++++++++++++++

By default, there are three active augmentations with preset parameters and example parameters. They can be seen here:

.. image:: images/augPipeline.png
  :width: 750
  :alt: Image showing the list of active augmentations (Gaussian Noise, JPEG Compression, Salt and Pepper)

You can reorder augmentations, delete them, or add them ("Add Augmentations" button).
To test different augmentations and different parameters, click the "Add Augmentations" button, and you will be presented with this window:

.. image:: images/augDialogExample.png
  :width: 750
  :alt: 

+++++++++++++++
Running Experiments
+++++++++++++++


Launching from Executable
-----------------

To launch the application, simply go to the main menu and click the "Noisey GUI" application shortcut.

Launching From Source
-----------------

To start the application: :code:`python3 app.py`
