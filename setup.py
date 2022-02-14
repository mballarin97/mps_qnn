import setuptools
import os
import os.path
import importlib.util

# Parse the version file
spec = importlib.util.spec_from_file_location("qcircha", "./qcircha/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

# Get the readme file
if os.path.isfile("README.md" ):
  with open("README.md", "r") as fh:
      long_description = fh.read()
else:
  long_description = ''

setuptools.setup(
    name = "qcircha",
    version = version_module.__version__,
    author = "Marco Ballarin, Stefano Mangini, Riccardo Mengoni",
    author_email = "marco97.ballarin@gmail.com",
    description = "Characterization of parametrized quantum circuits",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/mballarin97/MPS-QNN",
    project_urls = {},
    classifiers = [
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir = {
        "qcircha": "qcircha",
        "qcircha.entanglement" : "qcircha/entanglement",
        #"qcomps.qec" : "qcomps/qec",
    },
    #packages = setuptools.find_packages(where = "qcmps"),
    packages = ['qcircha', 'qcircha.entanglement'],#, 'qcomps.qec'],
    python_requires=">=3.6",
)
