from setuptools import setup, find_packages

setup(
    name="vistac_sdk",
    version="0.1.0",
    description="SDK for visual-tactile sensors usage, reconstruction, and calibration. Edited from https://github.com/joehjhuang/gs_sdk",
    author="Byung-Hyun Song",
    author_email="bh.song@snu.ac.kr",
    packages=find_packages(),
    install_requires=[
        "pillow==10.0.0",
        "numpy==1.26.4",
        "opencv-python>=4.9.0",
        "scipy>=1.13.1",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "PyYaml>=6.0.1",
        "matplotlib>=3.9.0",
        "ffmpeg-python",
        "nanogui",
        "open3d>=0.17.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        # Force estimation dependencies (Sparsh)
        "einops>=0.6",
        "timm>=0.9",
        "huggingface_hub>=0.19",
        "omegaconf>=2.3",
        "lightning>=2.0",
        "rich>=13.0",
    ],
    extras_require={
        # Optional CUDA acceleration for Sparsh attention kernels
        "gpu": [
            "xformers>=0.0.22",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "vistac-collect=vistac_sdk.calibration.collect_data:collect_data",
            "vistac-label=vistac_sdk.calibration.label_data:label_data",
            "vistac-prepare=vistac_sdk.calibration.prepare_data:prepare_data",
            "vistac-train=vistac_sdk.calibration.train_model:train_model",
            "vistac-test=vistac_sdk.calibration.test_model:test_model",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
