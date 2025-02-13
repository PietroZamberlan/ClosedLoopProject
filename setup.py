from setuptools import setup, find_packages

setup(
    name='closedloopproject',
    version='0.1.0',
    packages=find_packages(include=['config', 'config.*', 
                                    'src', 'src.*',
                                    'Gaussian-Processes.Spatial_GP_repo', 'Gaussian-Processes.Spatial_GP_repo*',]),
    package_dir={"":"." },
    include_package_data=True,
    install_requires=[
        # List your dependencies here
    ],
)

'''

You can adjust your setup.py so that it tells setuptools to look for packages in the entire repo root. 
This lets you install and reference packages from directories like config/, src/, and Gaussian-Processes/. 
One way to do this is to set the repository root as the package root by using the mapping

  package_dir = { "": "." }

and listing all the package patterns in the include option of find_packages. For example:



After installing the package (for example, with pip install -e . from the repo root), 
you can import modules from these directories without errors. 
Make sure each folder you wish to import from has an __init__.py file.
'''