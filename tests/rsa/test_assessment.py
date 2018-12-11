"""
Tests for rsa.assessment.RSA_Assessment class
"""


def test_projection_list(assessment_logic):
    # assert that this assessment can identify both frontal and lateral projections
    assert sorted(["frontal", "lateral"]) == assessment_logic.projection_list()


def test_scene_segment_list(assessment_logic):
    # assert that this assessment can identify these four segments
    assert sorted(assessment_logic.scene_segment_list()) == \
           sorted([
               "acetabular beads",
               "liner beads",
               "stem beads",
               "femoral beads"
           ])

## DONT KNOW WHAT THIS DOES, PLEASE ADD DOCUMENTATION
# def test_scene_segment_nogroup(default_dataset, assessment_logic):
#    ss = assessment_logic.scene_segment("acetabular beads")
#    ss.match_crossing_lines()
#    assert len(ss._idict["points"]) == np.min([len(ss.lines_a), len(ss.lines_b)])
#    assert ss.mse < 0.350, "MSE higher than expected for testing data"
#    default_dataset.flush()
