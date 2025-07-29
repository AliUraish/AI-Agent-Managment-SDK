from setuptools import setup, find_packages

setup(
    name="AI-HR-Agent",
    version="1.2.1",
    packages=find_packages(),
    install_requires=[
        'requests>=2.28.0',
        'aiohttp>=3.8.0',
        'python-dateutil>=2.8.0'
    ],
    author="Ali Uraish",
    author_email="aliuraish@gmail.com",
    description="A Python SDK for tracking AI Agent operations and metrics",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AliUraish/AI-Agent-Managment-SDK",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
) 