import sys
from pathlib import Path

import numpy as np

src_dir = Path(__file__).parent.joinpath("../../src").resolve()
sys.path.insert(0, str(src_dir))

from rsa.projection import BeadCenterAdjuster


def test_image(image_logic):
    # assert that the image logic class can correctly load these properties
    assert image_logic.assessment_name == "dx0"
    assert image_logic.name == "frontal"


def test_image_segment_list(image_logic):
    # assert that the image logic can identify these segments
    assert sorted(image_logic.image_segment_list()) == \
           sorted([
               "acetabular beads",
               "liner beads",
               "stem beads",
               "femoral beads"
           ])


def test_left_acetabulum_image_segments(image_logic):
    # assert that the image logic can load a segment, and verify this segment's name, number of coordinates, and type
    acet = image_logic.image_segment("acetabular beads")

    assert acet.name == "acetabular beads"
    assert np.array(acet.points).shape == (7, 2)
    assert acet._type == "beads"


def test_SincBeadCenterAdjuster(image_logic):
    detector = BeadCenterAdjuster.factory(image_logic.image, 413, 933, BeadCenterAdjuster.METHOD_SINC)
    print(detector.offset)
