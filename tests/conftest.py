"""Pytest bootstrap: put the repo root on sys.path so tests can import the
project packages (utils, calibration, preprocessing, training) without an
installed package. Keeps the suite runnable with a bare `pytest` from root.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
