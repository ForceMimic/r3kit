from setuptools import setup, find_packages

setup(
    name = 'r3kit', 
    version = '0.0.1', 
    license = 'MIT', 
    description = 'Research kits for real robots', 
    author = "Junbo Wang", 
    author_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    maintainer = "Junbo Wang", 
    maintainer_email = "sjtuwjb3589635689@sjtu.edu.cn", 
    url = "https://github.com/ForceMimic/r3kit", 
    packages = find_packages(), 
    include_package_data = True, 
    install_requires = [
        'numpy<2.0.0', 
        'opencv-python>=4.10.0', 
        'open3d>=0.18.0', 
        'matplotlib', 
        'scipy', 
        'transformations', 
        'pyrealsense2==2.53.1.4623', # needed by camera realsense
        'pyserial', # needed by encoder pdcd angle
    ], 
    zip_safe = False
)
