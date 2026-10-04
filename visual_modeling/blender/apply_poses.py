def apply_poses(ship, camera, T_world_deck, T_deck_camera):
    from mathutils import Matrix
    import numpy as np
    from navigation.common.frames import opencv_to_blender

    ship.matrix_world = Matrix(np.asarray(T_world_deck).tolist())
    camera.matrix_world = Matrix(
        (
            np.asarray(T_world_deck) @ np.asarray(T_deck_camera) @ opencv_to_blender()
        ).tolist()
    )
