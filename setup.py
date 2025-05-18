from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dropwise-metrics",
    version="0.1.1",
    author="Aryan Patil",
    author_email="aryanator01@gmail.com",
    description="TorchMetrics-compatible predictive uncertainty metrics using MC Dropout",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aryanator/dropwise-metrics",  # ✅ Change to match your new repo when ready
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchmetrics",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',
)
