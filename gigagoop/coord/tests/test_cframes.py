import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gigagoop.viz import SpaceGraph
from gigagoop.coord import get_world_coordinate_system, get_camera_coordinate_system, Transform

plt.ion()
matplotlib.use('Qt5Agg')

SHOW_FIGURES = False


def test_world_coordinate_systems():

    # Create all the frames
    M_WCS_WCS = get_world_coordinate_system('gg')
    M_UEWCS_WCS = get_world_coordinate_system('ue')
    M_BLWCS_WCS = get_world_coordinate_system('blender')
    M_GLWCS_WCS = get_world_coordinate_system('opengl')
    M_ARWCS_WCS = get_world_coordinate_system('arkit')

    # From the presentation, we have obvious equality in some of these frames...
    assert M_WCS_WCS == Transform()
    assert M_WCS_WCS == M_UEWCS_WCS
    assert M_GLWCS_WCS == M_ARWCS_WCS

    # Generate the nice figure
    shift = lambda M, s: Transform.from_rotation_and_origin(M.rotation, origin=[0, s, 0])

    if SHOW_FIGURES:
        sg = SpaceGraph(show_wcs=False)
        sg.add_cframe(shift(M_GLWCS_WCS, 0))
        sg.add_cframe(shift(M_BLWCS_WCS, 2))
        sg.add_cframe(shift(M_WCS_WCS, 4))

    # For a simple test that is not exhaustive, but probably good enough, let's define a point in our world system
    p_wcs = np.array([0.3, 0.4, 0.5])

    # Visually inspecting, it is clear that this point should be
    M_WCS_UEWCS = M_UEWCS_WCS.inverse
    M_WCS_BLWCS = M_BLWCS_WCS.inverse
    M_WCS_GLWCS = M_GLWCS_WCS.inverse
    M_WCS_ARWCS = M_ARWCS_WCS.inverse

    assert np.allclose(M_WCS_UEWCS(p_wcs), p_wcs)
    assert np.allclose(M_WCS_BLWCS(p_wcs), [0.4, 0.3, 0.5])
    assert np.allclose(M_WCS_GLWCS(p_wcs), [0.4, 0.5, -0.3])
    assert np.allclose(M_WCS_ARWCS(p_wcs), [0.4, 0.5, -0.3])


def test_camera_coordinate_system():
    """
    In addition to testing `get_camera_coordinate_system`, this test will also recreate the
    "Camera Coordinate System Conventions" slide in `20230628_cframes.pptx`.
    """
    M_CAM_CAM = get_camera_coordinate_system('gg')
    M_UECAM_CAM = get_camera_coordinate_system('ue')
    M_BLCAM_CAM = get_camera_coordinate_system('blender')
    M_GLCAM_CAM = get_camera_coordinate_system('opengl')
    M_ARCAM_CAM = get_camera_coordinate_system('arkit')
    M_CVCAM_CAM = get_camera_coordinate_system('opencv')

    assert M_CAM_CAM == Transform()
    assert M_CAM_CAM == M_UECAM_CAM
    assert M_BLCAM_CAM == M_GLCAM_CAM
    assert M_BLCAM_CAM == M_ARCAM_CAM

    M_CAM_WCS = Transform()

    if SHOW_FIGURES:
        M_GLCAM_WCS = M_CAM_WCS * M_GLCAM_CAM
        M_CVCAM_WCS = M_CAM_WCS * M_CVCAM_CAM

        shift = lambda M, s: Transform.from_rotation_and_origin(M.rotation, origin=[0, s, 0])

        sg = SpaceGraph(show_wcs=False)
        sg.add_cframe(shift(M_CVCAM_WCS, 0))
        sg.add_cframe(shift(M_GLCAM_WCS, 2))
        sg.add_cframe(shift(M_CAM_WCS, 4))

    M_CAM_UECAM = M_UECAM_CAM.inverse
    M_CAM_ARCAM = M_ARCAM_CAM.inverse
    M_CAM_BLCAM = M_BLCAM_CAM.inverse
    M_CAM_GLCAM = M_GLCAM_CAM.inverse
    M_CAM_CVCAM = M_CVCAM_CAM.inverse

    p_cam = np.array([0.3, 0.4, 0.5])
    assert np.allclose(M_CAM_UECAM(p_cam), p_cam)
    assert np.allclose(M_CAM_ARCAM(p_cam), [0.4, 0.5, -0.3])
    assert np.allclose(M_CAM_BLCAM(p_cam), [0.4, 0.5, -0.3])
    assert np.allclose(M_CAM_GLCAM(p_cam), [0.4, 0.5, -0.3])
    assert np.allclose(M_CAM_CVCAM(p_cam), [0.4, -0.5, 0.3])


if __name__ == '__main__':
    test_world_coordinate_systems()
    test_camera_coordinate_system()
