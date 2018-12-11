# EOS-RSA: Röntgen Stereogrammetry Analysis using the EOS Imaging System


**Authors:** [Kristian Kjærgaard](http://findresearcher.sdu.dk:8080/portal/da/person/kkjaergaard), Charles Bragdon, Thiusius Rajeeth Savarimuthu, Ming Ding.


This project allows for marker-based RSA analyses for measuring micro-motion of prosthetic implants using the EOS Imaging System.

This software is built as the minimal viable product for this purpose and is intended for people who are familiar with both RSA and Python/YAML/using the terminal.


## Features

**Supported segments:** Point clouds.

**Accuracy and precision:** The mean error ± standard deviation is between 2.3 ± 5.2 μm and 10.8 ± 59.1 μm.

**Extendable:** As published under the General Public License version 3, you are free to modify, extend, or improve this software under the terms described in the license.


## Using the software

At the moment this software does not have a usable user manual, so please see the scrips in `bin/` and the project file example in `data/hip_phantom_simple/yaml/delta x at isocenter.yaml`.

If you use this software, please cite it as: (TBP).


## System requirements

Python 3 with the following packages: Numpy, Scipy, PyQtGraph, PyQt5, click, OpenCV, visvis, PyDicom, yaml, scikit-learn. To run the notebooks, you also need Jupyter, Matplotlib, Pandas, and Numba.


## License terms

EOS-RSA: Röntgen Stereogrammetry Analysis using the EOS Imaging System. Copyright © 2018  Kristian Kjærgaard

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
