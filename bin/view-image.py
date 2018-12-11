#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

import click
from PyQt5.QtWidgets import QApplication

src_dir = Path(__file__).parent.joinpath("../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.dataset import RSA_Dataset
from rsa.series import RSA_Series_Logic
from windows.ImageWindow import ImageWindow


@click.command()
@click.argument("filename", nargs=1, type=click.Path(exists=True))
def view_image(filename):
    series = RSA_Series_Logic(RSA_Dataset(filename))

    app = QApplication(sys.argv)
    imageWindow = ImageWindow(series)
    imageWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    view_image()
