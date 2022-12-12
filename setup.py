"""
Run the build process by running the command 'python setup.py build'
If everything works well you should find a subdirectory in the build
subdirectory that contains the files needed to run the application.
"""

from cx_Freeze import Executable, setup


executables = [Executable("app.py")]

options = {
    "build_exe": {
        # exclude packages that are not really needed
        "excludes": [
            "tkinter",
            "unittest",
            "email",
            "http",
            "xml",
            "pydoc",
        ],
        'packages': ['cv2'],
        "include_files": ["../Noisey-image/imgs/default_imgs/"],
    }
}

setup(
    name="Noisey Image",
    version="0.1",
    description="Add augmentations and run models",
    options=options,
    executables=executables,
)
