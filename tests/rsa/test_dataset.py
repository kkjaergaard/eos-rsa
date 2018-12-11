from pathlib import Path

import numpy as np
import pydicom as dcm


def test_load_frontal_image(default_dataset):
    assert np.allclose(
        default_dataset.load_dicom("assessments.[name=dx0].projections.[name=frontal].dicom_path").pixel_array,
        dcm.dcmread(
            str(Path(__file__).parent.joinpath("../../data/hip_phantom_sample/dicom/DX000000.dcm"))
        ).pixel_array
    )


# def test_regex_key_string(default_dataset):
#    assert default_dataset.get("assessments.baseline.images.frontal.image_path") == \
#           default_dataset.get("assessments.baseline.images.[0].image_path")


# # we only perform one dicom load test as it is time consuming
# def test_load_lateral_image(default_dataset):
#     assert np.allclose(
#         default_dataset.load_dicom("assessments.[name=baseline].images.[name=lateral].dicom_path").pixel_array,
#         dcm.dcmread(
#             str(Path(__file__).parent.joinpath("lateral.dcm"))
#         ).pixel_array
#     )
