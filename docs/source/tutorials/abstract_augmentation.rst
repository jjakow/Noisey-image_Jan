Integrating New Augmentation
==========================

In this tutorial, we are going to look at how to integrate a new augmentation method into the GUI interface.

General Overview
-----------------
In terms of a general overview of how the augmentation system works in the GUI interface, it is broken down into three components:

- Augmentation (abstract class)
- AugmentationPipeline
- AugDialog (QDialog window)

.. image:: images/current_gui_workflow.png
  :width: 750
  :alt: Diagram showing each one of these functions are wrapped around an abstract class called an ”Augmentation”. This abstract class allows all functions to integrate into the ”AugmentationPipeline” class, which controls the execution and order of all chosen augmentations.

To start the process, you need to develop your "core" augmentation function that is responsible for modifying the image. Additionally, the augmentation function requires a single parameter that changes the behavior of the modification function.

Augmentation
-----------------
The :code:`Augmentation` class is an abstract class that is very similar to how the :code:`Model` class behaves. It acts as a wrapper around the core augmentation function that you are responsible for writing. With this wrapper, the new augmentation function can be neatly placed within the AugmentationPipeline.

AugmentationPipeline
-----------------

AugDialog
-----------------



Example
-----------------

Let's do an example! We will be implementing :code:`Gaussian Noise`