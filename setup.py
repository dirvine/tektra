#!/usr/bin/env python3
"""
Setup script for Tektra AI Assistant.

This script creates a Python package that can be easily installed via pip
and includes all necessary components for the AI assistant.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="tektra",
    version="0.1.0",
    author="Tektra Team",
    author_email="contact@tektra.ai",
    description="Advanced AI assistant with voice, vision, and robotics capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tektra/tektra",
    packages=find_packages(where="backend"),
    package_dir={"": "backend"},
    include_package_data=True,
    package_data={
        "tektra": [
            "frontend/build/**/*",
            "config/*",
            "data/*",
            "models/*",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0",
        "asyncpg>=0.29.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        "python-multipart>=0.0.6",
        "websockets>=12.0",
        "python-dotenv>=1.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "httpx>=0.25.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
        "ml": [
            "torch>=2.1.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "optimum>=1.14.0",
        ],
        "audio": [
            "soundfile>=0.12.0",
            "librosa>=0.10.0",
            "pyaudio>=0.2.11",
        ],
        "vision": [
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0",
        ],
        "robotics": [
            "pyserial>=3.5",
            "pybullet>=3.2.5",
        ],
        "all": [
            "torch>=2.1.0",
            "transformers>=4.35.0",
            "accelerate>=0.24.0",
            "optimum>=1.14.0",
            "soundfile>=0.12.0",
            "librosa>=0.10.0",
            "pyaudio>=0.2.11",
            "opencv-python>=4.8.0",
            "mediapipe>=0.10.0",
            "pyserial>=3.5",
            "pybullet>=3.2.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "tektra=tektra.cli:main",
            "tektra-server=tektra.server:main",
            "tektra-setup=tektra.setup:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/tektra/tektra/issues",
        "Source": "https://github.com/tektra/tektra",
        "Documentation": "https://docs.tektra.ai",
        "Homepage": "https://tektra.ai",
    },
    keywords="ai assistant voice vision robotics ml chatbot",
    zip_safe=False,
)