from dataclasses import dataclass


@dataclass
class WindowParams:
    title: str = ''
    width: int = 1920
    height: int = 1080

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

#
# @dataclass
# class CameraParams:
#     width: int = 1920
#     height: int = 1080
#     fov_vertical: float = 60.0
#     near: float = 0.1
#     far: float = 100.0
#
#     @property
#     def aspect_ratio(self) -> float:
#         return self.width / self.height
#
#
# @dataclass
# class UECameraParams(CameraParams):
#     width: int = 1620
#     height: int = 1080
#     fov_vertical: float = 53.13010235415598
#     sensor_width: float = 36.0     # mm
#     sensor_height: float = 24.0    # mm
#     focal_length: float = 24.0     # mm
