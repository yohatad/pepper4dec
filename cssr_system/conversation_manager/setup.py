from setuptools import find_packages, setup
from glob import glob

package_name = 'conversation_manager'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),  # <-- ADD THIS LINE
    install_requires=['setuptools', 'openai', 'chromadb', 'sentence-transformers'],
    zip_safe=True,
    maintainer='Muhammed Danso and Yohannes Haile',
    maintainer_email='mahadanso79@gmail.com and yohatad000@gmail.com',
    description='RAG system server wrapper for large language model',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'conversation_manager = conversation_manager.conversation_manager_application:main',
        ],
    },
    
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
        # (f"share/{package_name}/models", glob("models/*")),
        (f"share/{package_name}/data", glob("data/*")),
    ],
)
