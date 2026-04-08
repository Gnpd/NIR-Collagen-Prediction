"""
Build script for PyInstaller.
Run from the api/ directory: python build.py
"""
import sys
import PyInstaller.__main__

sep = ";" if sys.platform == "win32" else ":"

PyInstaller.__main__.run([
    "main.py",
    "--onefile",
    "--name", "nir-collagen-api",
    f"--add-data=plsr_2045.json{sep}.",
    f"--add-data=rf_2045.json{sep}.",
    "--hidden-import=uvicorn.logging",
    "--hidden-import=uvicorn.loops",
    "--hidden-import=uvicorn.loops.auto",
    "--hidden-import=uvicorn.protocols",
    "--hidden-import=uvicorn.protocols.http",
    "--hidden-import=uvicorn.protocols.http.auto",
    "--hidden-import=uvicorn.protocols.websockets",
    "--hidden-import=uvicorn.protocols.websockets.auto",
    "--hidden-import=uvicorn.lifespan",
    "--hidden-import=uvicorn.lifespan.on",
    "--hidden-import=multipart",
    "--collect-all=chemotools",
    "--collect-all=openmodels",
])
