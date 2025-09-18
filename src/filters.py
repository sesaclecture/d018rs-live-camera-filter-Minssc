import cv2
import numpy as np


class Filters:
    # TODO: Image kernels
    Kernels:dict = {
        'Original': np.array([[0, 0, 0], [0, 1, 0], [0, 0 ,0]], dtype=np.float32),
        'Blur' : np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9,
        'Gaussian blur' : np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]], dtype=np.float32),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        'Sobel (X)': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        'Sobel (Y)': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        'Edge detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    }

    def __init__(self, kernels:dict=Kernels):
        self.kernels = kernels
        self.current_kernel = next(iter(kernels.keys()))

    def apply_filter(self, frame, filter_name) -> np.array:
        self.current_kernel = filter_name
        return cv2.filter2D(frame, -1, self.kernels[self.current_kernel])

    def get_current_filter_name(self) -> str:
        return self.current_kernel

    def switch_next_filter(self):
        klist = list(self.kernels.keys())
        idx = klist.index(self.current_kernel)
        self.current_kernel = klist[(idx + 1) % len(self.kernels)]

    def switch_previous_filter(self):
        klist = list(self.kernels.keys())
        idx = klist.index(self.current_kernel)
        self.current_kernel = klist[(idx - 1) % len(self.kernels)]
