#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

import click

src_dir = Path(__file__).parent.joinpath("../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.dataset import RSA_Dataset
from rsa.series import RSA_Series_Logic
from windows.SceneWindow import SceneWindow


# if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
#    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
#
# if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
#    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


@click.command()
@click.argument("filename", nargs=1, type=click.Path(exists=True))
@click.argument("assessment_name", nargs=1)
def view_scene(filename, assessment_name):
    series = RSA_Series_Logic(RSA_Dataset(filename))
    sceneWindow = SceneWindow(series.assessment(assessment_name))
    sceneWindow.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    view_scene()
