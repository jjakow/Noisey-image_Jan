Integrating New Augmentation
==========================

In this tutorial, we are going to look at how to integrate a new augmentation method into the GUI interface.

General Overview
-----------------
In terms of a general overview of how the augmentation system works in the GUI interface, it is broken down into three components:

- Augmentation (abstract class)
- AugmentationPipeline
- AugDialog (QDialog window)

.. image:: ../images/current_gui_workflow.png
  :width: 750
  :alt: Diagram showing each one of these functions are wrapped around an abstract class called an ”Augmentation”. This abstract class allows all functions to integrate into the ”AugmentationPipeline” class, which controls the execution and order of all chosen augmentations.

To start the process, you need to develop your "core" augmentation function that is responsible for modifying the image. Additionally, the augmentation function requires a single parameter that changes the behavior of the modification function.

Augmentation
-----------------
The :code:`Augmentation` class is an abstract class that is very similar to how the :code:`Model` class behaves. It acts as a wrapper around the core augmentation function that you are responsible for writing. With this wrapper, the new augmentation function can be neatly placed within the AugmentationPipeline.

AugmentationPipeline
-----------------

The :code:`AugmentationPipeline` is a class that manages multiple, active :code:`Augmentation` objects along with their order of execution and other metadata. This class also manages things, such as validity of all the :code:`Augmentation` objects' parameter validity. For the Augmentation GUI, a single :code:`AugmentationPipeline` object is declared in :code:`transforms.py`.

AugDialog
-----------------

The :code:`AugDialog` is an inherited :code:`QDialog` class that represents the functionality of the Augmentation dialog window.

Example
-----------------

Let's do an example! We will be implementing :code:`Gaussian Noise`. The first step is to define the core function that will take in a parameter (standard deviation) and an image that will be modified. This function is typically defined in the :code:`transform_funcs.py` file.

**Core Function**

.. code-block:: python
    :caption: Creating a core function that will apply Gaussian Noise to an image.

    def gaussian_noise(image:np.ndarray, std:int):
      mean = 2
      # only control standard dev:
      normal_matrix = np.random.normal(mean, std, size=image.shape)
      combined = image+normal_matrix
      np.clip(combined, 0, 255, out=combined)
      return combined.astype('uint8')

The next step is to create an :code:`Augmentation` object and place it into the :code:`augList` dictionary inside :code:`transforms.py`. You can also define a validity function that checks if a given paramter is within 

**Augmentation Object Creation**

.. code-block:: python
    :caption: Creating a function to check if value of the core function is valid or not.
    
    def __gaussianNoiseCheck__(param): return param > 0

.. code-block:: python
    :caption: Inserting the core function and the validity function into the augList variable as an entry. This will be automatically made into an Augmentation object and placed into the AugmentationPipeline.

    "Gaussian Noise": {"function": gaussian_noise, "default": [1,10,15,20,25,30,35,40,45,50,55,60], "example":25, "limits":trans.__gaussianNoiseCheck__}

.. code-block:: python
    :caption: This is what is essentially called when parsing through augList.

    augPosition = 0
    exampleParameter = 50
    defaultParameters = [1,10,20,30]
    augObject = Augmentation( ("Gaussian Noise", gaussianNoise), augPosition, (defaultParameters, exampleParameter), verbose=False, limit=__gaussianNoiseCheck__)