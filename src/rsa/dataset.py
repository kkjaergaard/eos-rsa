"""

RSA data stores and IO.

TODO: Change name to RSA_DataStore for clarity, change all member names from dataset to datastore for clarity
TODO: Implement ObjectPath access to the idict nested dictionary, add unit tests when adding this support
TODO: Adapt all classes to the new get method

"""

import logging
import re
from pathlib import Path

import cv2
import objectpath
import pydicom as dcm
import yaml


class RSA_Dataset(object):
    """
    RSA data store in the 3-tier pattern. Responsible for loading and saving data and making data available to logic classes.
    """

    def __init__(self, fname):
        """
        Instantiate RSA_Dataset

        Parameters
        ----------
        fname: string
            Filename of RSA dataset
        """

        self.fpath = Path(fname).resolve()
        f = open(self.fpath, "r")
        self.idict = yaml.load(f.read())
        self.tree = objectpath.Tree(self.idict)
        f.close()

        self.key_pattern = re.compile(r'(\[(\w+)=([\w /]+)\])|(\[(\d+)\])|(\w+)')

        self.logger = logging.getLogger(__name__)

    def get(self, key_string):
        """
        Return items in nested dicts using strings. E.g. obj.get("a.b.c") instead of obj["a"]["b"]["c"].

        Only works for key-value pairs, not for lists.
        """

        idict = self.idict
        for key in key_string.split("."):

            key_match = self.key_pattern.match(key)
            assert key_match, "Invalid key '{}'".format(key)

            if key_match.group(1):
                keyval_match = False
                for l in idict:
                    if l[key_match.group(2)] == key_match.group(3):
                        idict = l
                        keyval_match = True
                        break
                if not keyval_match:
                    raise KeyError
            elif key_match.group(5):
                idict = idict[int(key_match.group(5))]
            elif key_match.group(6):
                idict = idict[key_match.group(6)]

            # if key_match.group("key"):
            #    idict = idict[key_match.group("key")]
            # if key_match.group("index"):
            #    key = list(idict.keys())[int(key_match.group("index"))]
            #    idict = idict[key]
        return idict

    def load_image(self, key_string, flags=cv2.IMREAD_GRAYSCALE):
        """
        Load and return image data. For DICOM images only the pixel data is returned.

        Parameters
        ----------
        key_string: string
            Key string for yaml item
        flags: int
            Flags for cv2.imread() function

        Returns
        -------
        ndarray
        """
        image_path = self.fpath.parent.joinpath(self.get(key_string))
        assert image_path.is_file(), "Image path {} does not exist".format(image_path)
        return cv2.imread(str(image_path), flags)

    def load_dicom(self, key_string):
        image_path = self.fpath.parent.joinpath(self.get(key_string))
        self.logger.debug("loading", image_path)
        assert image_path.is_file(), "Dicom path {} does not exist".format(image_path)
        return dcm.dcmread(str(image_path))

    def load_image_headers(self, key_string):
        raise NotImplementedError

    def _set(self, p, v):
        raise NotImplementedError

    # @classmethod
    # def from_file(cls, fname):
    #     """
    #     Load a yaml dataset from a file.
    #
    #     Parameters
    #     ----------
    #     fname: string
    #         Filename of yaml to load into dataset object.
    #
    #     Returns
    #     -------
    #     RSA_Dataset object from file
    #     """
    #     fpath = Path(fname).resolve()
    #     f = open(fpath, "r")
    #     rsa_dataset = cls(yaml.load(f.read()), dir_prefix=fpath.parent)
    #     f.close()
    #     return rsa_dataset

    def flush(self, **kwargs):
        self.fpath = kwargs.get("fname", self.fpath)
        f = open(self.fpath, "w")
        f.write(yaml.dump(self.idict))
        f.close()
