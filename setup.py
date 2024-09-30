from setuptools import setup, find_packages

setup(
    name="flash_attention_mlx",
    version="0.1.0",
    description="Flash Attention implementation using the MLX framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jckwind",
    author_email="jckwind11@gmail.com",
    url="https://github.com/Jckwind/MLX-FlashAttention",
    packages=find_packages(),
    install_requires=[
        "mlx",  # Assuming MLX is a package
        "torch",  # If you're using PyTorch
        "numpy",  # If you're using NumPy
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
