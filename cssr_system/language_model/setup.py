from setuptools import find_packages, setup

package_name = 'language_model'

setup(
    name=package_name,
    version='0.0.0',
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'openai', 'chromadb', 'sentence-transformers'],
    zip_safe=True,
    maintainer='Muhammed Danso',
    maintainer_email='mahadanso79@gmail.com',
    description='RAG system server wrapper for large language model',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'service = language_model.rag_application:main',
        ],
    },
    
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{pkg}"]),
        (f"share/{pkg}", ["package.xml"]),
        # (f"share/{pkg}/launch", glob("launch/*.launch.py")),
        (f"share/{pkg}/config", glob("config/*")),
        # (f"share/{pkg}/models", glob("models/*")),
        (f"share/{pkg}/data", glob("data/*")),
    ],
)
