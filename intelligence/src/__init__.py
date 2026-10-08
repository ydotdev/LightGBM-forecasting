import os

# Prevent OpenBLAS memory allocation failures on Windows with Python 3.14+
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
