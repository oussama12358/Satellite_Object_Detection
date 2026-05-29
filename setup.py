from setuptools import setup, find_packages

setup(
    name="satdet",
    version="1.0.0",
    description="Satellite Object Detection System using YOLOv8",
    author="SatDet Team",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "ultralytics>=8.0.196",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "loguru>=0.7.2",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
        ],
        "full": [
            "mlflow>=2.8.0",
            "streamlit>=1.28.0",
            "onnx>=1.15.0",
            "onnxruntime-gpu>=1.16.0",
            "sahi>=0.11.14",
        ],
    },
    entry_points={
        "console_scripts": [
            "satdet-train=src.training.trainer:main",
            "satdet-eval=src.evaluation.evaluator:main",
            "satdet-predict=src.inference.batch_predictor:main",
            "satdet-export=src.models.onnx_export:main",
            "satdet-serve=src.api.main:main",
        ],
    },
)
