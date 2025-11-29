from setuptools import setup
import os
from glob import glob

package_name = 'custom_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lab',
    maintainer_email='lab@todo.todo',
    description='Custom launch files',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'color_compressor = custom_launch.color_compressor:main',  # Add this line
            'depth_roi_service = custom_launch.depth_roi_service:main'
        ],
    },
)