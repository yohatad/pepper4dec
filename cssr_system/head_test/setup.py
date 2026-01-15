from setuptools import find_packages, setup

package_name = 'head_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoha',
    maintainer_email='yohatad000@gmail.com',
    description='Head reactivity test for Pepper robot: measures response time to random image points',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'head_reactivity_test = head_test.head_reactivity_test:main',
        ],
    },
)
