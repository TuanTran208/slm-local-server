"""
PyInstaller script to create an executable
Run with: python package.py
"""
import PyInstaller.__main__
import os
import shutil


def package_app():
    # Create a directory for distribution files
    os.makedirs("dist", exist_ok=True)

    # Use PyInstaller to create the executable
    PyInstaller.__main__.run([
        'service.py',  # Script to be converted
        '--name=SLM_Server',  # Name of the executable
        '--onedir',  # Create a directory with all dependencies
        '--clean',  # Clean PyInstaller cache
        '--add-data=model_files;model_files',  # Include model files
    ])

    print("Executable created in ./dist/SLM_Server/")
    print("To run: ./dist/SLM_Server/SLM_Server.exe")


if __name__ == "__main__":
    package_app()