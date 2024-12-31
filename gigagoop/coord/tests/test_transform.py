import numpy as np

from gigagoop.coord import Transform, Translate, RotateX, RotateY, RotateZ, GazeAt


def test_initialize_transform():
    # NO INPUTS
    # =========
    # With no inputs, we should get back identity
    T = Transform()

    assert np.all(T.matrix == np.eye(4))

    # We can pull out various components of this matrix as
    assert np.all(T.x_axis == [1, 0, 0])
    assert np.all(T.y_axis == [0, 1, 0])
    assert np.all(T.z_axis == [0, 0, 1])
    assert np.all(T.origin == [0, 0, 0])
    assert np.all(T.rotation == np.eye(3))

    # MATRIX
    # ======
    # We can initialize the transform directly with a 4x4 matrix
    M = [[0, -1, 0, 0],
         [1,  0, 0, 0],
         [0,  0, 1, 0],
         [0,  0, 0, 1]]

    T = Transform(M)                   # ordered input
    assert T == Transform(matrix=M)    # named input

    # The matrix above rotates about Z 90 degrees
    R = RotateZ(90).matrix[:3, :3]
    assert np.all(np.isclose(T.rotation, R))

    # ROTATION
    # ========
    # We can equivalently initialize it as
    T = Transform.from_rotation(R)
    assert np.all(T.origin == [0, 0, 0]) and np.all(np.isclose(T.rotation, R))

    # ORIGIN
    # ======
    T = Transform.from_origin([1, 2, 3])
    assert np.all(np.isclose(T.rotation, np.eye(3)))

    # ROTATION AND ORIGIN
    # ===================
    T = Transform.from_rotation_and_origin(R, [1, 2, 3])
    assert np.all(np.isclose(T.matrix, [[0, -1, 0, 1],
                                        [1,  0, 0, 2],
                                        [0,  0, 1, 3],
                                        [0,  0, 0, 1]]))

    # INVERSE
    # =======
    R = RotateZ(45).matrix[:3, :3]
    M = np.r_[np.c_[R, [1, 2, 3]], [[0, 0, 0, 1]]]
    T = Transform(M)
    assert not T.has_scale
    assert np.all(np.isclose(T.inverse.matrix, np.linalg.inv(M)))

    R = RotateZ(45).matrix[:3, :3]
    R[0, 1] *= 2
    M = np.r_[np.c_[R, [1, 2, 3]], [[0, 0, 0, 1]]]
    T = Transform(M)
    assert T.has_scale
    assert np.all(np.isclose(T.inverse.matrix, np.linalg.inv(M)))


def test_initialize_translate():
    T = Translate()
    assert np.all(T.matrix == np.eye(4))

    T = Translate(1, 2, 3)
    assert np.all(T.rotation == np.eye(3)) and np.all(T.origin == [1, 2, 3])

    assert T == Translate(dx=1, dy=2, dz=3)


def test_initialize_rotations():
    assert np.all(RotateX().matrix == np.eye(4))
    assert np.all(RotateY().matrix == np.eye(4))
    assert np.all(RotateZ().matrix == np.eye(4))

    # The set rotation matrices match the Unreal Engine convention...

    # If you rotate a vector 45 deg in a plane, you get this magic number ;-P
    m = np.sqrt(2)/2

    # ROTATE X
    # ========
    T = RotateX(-45)
    assert np.all(np.isclose(T.x_axis, [1, 0, 0]))
    assert np.all(np.isclose(T.y_axis, [0, m, m]))
    assert np.all(np.isclose(T.z_axis, [0, -m, m]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))

    T = RotateX(45)
    assert np.all(np.isclose(T.x_axis, [1, 0, 0]))
    assert np.all(np.isclose(T.y_axis, [0, m, -m]))
    assert np.all(np.isclose(T.z_axis, [0, m, m]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))

    # ROTATE Y
    # ========
    T = RotateY(-45)
    assert np.all(np.isclose(T.x_axis, [m, 0, -m]))
    assert np.all(np.isclose(T.y_axis, [0, 1, 0]))
    assert np.all(np.isclose(T.z_axis, [m, 0, m]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))

    T = RotateY(45)
    assert np.all(np.isclose(T.x_axis, [m, 0, m]))
    assert np.all(np.isclose(T.y_axis, [0, 1, 0]))
    assert np.all(np.isclose(T.z_axis, [-m, 0, m]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))

    # ROTATE Z
    # ========
    T = RotateZ(-45)
    assert np.all(np.isclose(T.x_axis, [m, -m, 0]))
    assert np.all(np.isclose(T.y_axis, [m, m, 0]))
    assert np.all(np.isclose(T.z_axis, [0, 0, 1]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))

    T = RotateZ(45)
    assert np.all(np.isclose(T.x_axis, [m, m, 0]))
    assert np.all(np.isclose(T.y_axis, [-m, m, 0]))
    assert np.all(np.isclose(T.z_axis, [0, 0, 1]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))


def test_applying_transforms():

    # TRANSLATE
    # =========
    # Translate a bunch of points to a new origin
    T = Translate(1, 2, 3)

    old_points = [[0, 0, 0],
                  [1, 1, 1]]

    new_points = [[1, 2, 3],
                  [2, 3, 4]]

    assert np.all(new_points == T(old_points))     # functional form
    assert np.all(new_points == T * old_points)    # standard multiply
    assert np.all(new_points == T @ old_points)    # matrix multiply

    # ROTATION
    # ========
    # A vector at `[1, 0, 0]` with an angle of 90 degrees should rotate to `[0, 1, 0]` - verify this quickly with scipy
    R = RotateZ(90).matrix[:3, :3]
    x = [1, 0, 0]
    y = R @ x
    assert np.all(np.isclose(y, [0, 1, 0]))

    # Now make sure our transform library does the same
    y = RotateZ(90) @ x
    assert np.all(np.isclose(y, [0, 1, 0]))

    # INTERESTING EXAMPLE
    # ===================
    # Consider a more interesting example where we use our "standardized notation" ;-). We start by defining two
    # coordinate systems
    #     WCS (World Coordinate System)
    #         This will represent the coordinate system that points live in before the transform
    #     OBJ (Object Coordinate System
    #         The coordinate system points live in after the transform

    # Define an object frame `M_OBJ_WCS` that consists of:
    #    1) rotate it 90 degrees so now the new X axis is the old Y axis
    #    2) translate it along the X axis by some amount
    # which means
    #     M_OBJ_WCS
    #         OBJ coordinate frame defined w.r.t. WCS
    #     M_OBJ_WCS.inverse = M_WCS_OBJ
    #         WCS to OBJ transform
    #         WCS coordinate frame defined w.r.t. OBJ coordinates
    M_OBJ_WCS = Translate(dx=10, dy=0, dz=0) * RotateZ(90)

    # Now from inspection...
    p_obj = [1, 0, 0]
    p_wcs = M_OBJ_WCS @ p_obj
    assert np.all(np.isclose(p_wcs, [10, 1, 0]))

    # Now the inverse should recover the original point - pick your favorite form
    assert np.all(np.isclose(M_OBJ_WCS.inverse * p_wcs, p_obj))
    assert np.all(np.isclose(M_OBJ_WCS.inverse @ p_wcs, p_obj))
    assert np.all(np.isclose(M_OBJ_WCS.inverse(p_wcs), p_obj))


def test_from_methods():
    # Transform.from_rotation
    R = (RotateX(10) * RotateY(-5) * RotateZ(3)).matrix[:3, :3]
    T = Transform.from_rotation(R)
    assert np.all(np.isclose(T.rotation, R))
    assert np.all(np.isclose(T.origin, np.zeros(3)))

    # Transform.from_origin
    origin = [1, 2, 3]
    T = Transform.from_origin(origin)
    assert np.all(np.isclose(T.rotation, np.eye(3)))
    assert np.all(np.isclose(T.origin, origin))

    # Transform.from_rotation_and_origin
    T = Transform.from_rotation_and_origin(R, origin)
    assert np.all(np.isclose(T.rotation, R))
    assert np.all(np.isclose(T.origin, origin))

    # Transform.from_unreal
    # ---------------------
    T = Transform.from_unreal(rotation=[90, 90, 90])
    assert np.all(np.isclose(T.x_axis, [0, 0, 1]))
    assert np.all(np.isclose(T.y_axis, [0, 1, 0]))
    assert np.all(np.isclose(T.z_axis, [-1, 0, 0]))
    assert np.all(np.isclose(T.origin, [0, 0, 0]))

    T = Transform.from_unreal(rotation=[90, 90, 90], location=[1, 2, 3])
    assert np.all(np.isclose(T.x_axis, [0, 0, 1]))
    assert np.all(np.isclose(T.y_axis, [0, 1, 0]))
    assert np.all(np.isclose(T.z_axis, [-1, 0, 0]))
    assert np.all(np.isclose(T.origin, [1, 2, 3]))

    T = Transform.from_unreal(rotation=[90, 90, 90], location=[1, 2, 3], scale=[0.5, 0.5, 0.5])
    assert np.all(np.isclose(T.x_axis, [0, 0, 0.5]))
    assert np.all(np.isclose(T.y_axis, [0, 0.5, 0]))
    assert np.all(np.isclose(T.z_axis, [-0.5, 0, 0]))
    assert np.all(np.isclose(T.origin, [1, 2, 3]))

    # Transform.from_lookat
    # ---------------------
    try:

        # Example 1
        # .........
        # The standard `LookAt` transform defined in OpenGL places the camera at "eye", points it at "target", and
        # orients the roll with help from the "up" vector. The returned transforms is then a mapping from `WCS` to `CAM`
        # and *not* the camera pose w.r.t. WCS.
        import glm

        eye = [0, 0, 1]
        target = [0, 0, 0]
        up = [0, 1, 0]
        G_WCS_CAM = glm.lookAt(eye, target, up)

        # Take note that `G_WCS_CAM` is a transformation from `WCS` to `CAM`. The camera w.r.t. WCS is given as
        G_CAM_WCS = Transform(np.asarray(G_WCS_CAM)).inverse

        # Example 2
        # .........
        eye = [0, 1, 1]
        target = [0, 0, 0]
        up = [0, 1, 0]
        G_CAM_WCS = glm.inverse(glm.lookAt(eye, target, up))

    except:
        pass

    # Example 1
    # .........
    M_CAM_WCS = GazeAt(eye=[-1, 0, 0], target=[0, 0, 0], up=[0, 0, 1])
    assert np.all(np.isclose(M_CAM_WCS.matrix, [[1, 0, 0, -1],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]]))

    # Example 2
    # .........
    m = np.sqrt(2)/2
    M_CAM_WCS = GazeAt(eye=[-1, 0, 1], target=[0, 0, 0], up=[0, 0, 1])
    assert np.all(np.isclose(M_CAM_WCS.matrix, [[m,  0, m, -1],
                                                [0,  1, 0, 0],
                                                [-m, 0, m, 1],
                                                [0,  0, 0, 1]]))


def test_to_methods():
    T = Transform.from_unreal(rotation=[10, 0, 0])
    _, rotation = T.to_unreal()
    assert np.all(np.isclose(rotation, [10, 0, 0]))

    T = Transform.from_unreal(rotation=[0, 10, 0])
    _, rotation = T.to_unreal()
    assert np.all(np.isclose(rotation, [0, 10, 0]))

    T = Transform.from_unreal(rotation=[0, 0, 10])
    _, rotation = T.to_unreal()
    assert np.all(np.isclose(rotation, [0, 0, 10]))

    T = Transform.from_unreal(rotation=[10, 20, 30])
    _, rotation = T.to_unreal()
    assert np.all(np.isclose(rotation, [10, 20, 30]))


if __name__ == '__main__':
    test_initialize_transform()
    test_initialize_translate()
    test_initialize_rotations()
    test_applying_transforms()
    test_from_methods()
    test_to_methods()
