import sys
from pathlib import Path

import numpy as np
import pytest

src_dir = Path(__file__).parent.joinpath("../../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.dataset import RSA_Dataset
from rsa.series import RSA_Series_Logic, RSA_Scene_Segment_Motion


@pytest.fixture
def default_dataset():
    return RSA_Dataset(
        str(Path(__file__).parent.joinpath("../../data/hip_phantom_sample/yaml/delta x at isocenter.yaml")))


@pytest.fixture
def series_logic(default_dataset):
    return RSA_Series_Logic(default_dataset)


@pytest.fixture
def assessment_logic(series_logic):
    return series_logic.assessment("dx0")


@pytest.fixture
def image_logic(assessment_logic):
    return assessment_logic.projection("frontal")

@pytest.fixture
def default_pointset():
    return np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 1],
    ])

@pytest.fixture
def scene_segment_motion():
    rsa_series = RSA_Series_Logic(
        RSA_Dataset(str(Path(__file__).parent.joinpath("../../data/hip_phantom_sample/yaml/delta x at isocenter.yaml")))
    )
    motion = RSA_Scene_Segment_Motion()
    assessment_names = rsa_series.assessment_list()
    motion._from = rsa_series.assessment(assessment_names[0])
    motion.to = rsa_series.assessment(assessment_names[1])
    motion.refsegment_name = "acetabular beads"
    motion.segment_name = "femoral beads"
    return motion
