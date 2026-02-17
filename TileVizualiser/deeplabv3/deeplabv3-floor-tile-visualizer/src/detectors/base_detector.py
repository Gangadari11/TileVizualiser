from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def load_model(self, model_path):
        """Load the detection model from the specified path."""
        pass

    @abstractmethod
    def detect_floor(self, image):
        """Detect the floor in the provided image and return a mask."""
        pass

    @abstractmethod
    def preprocess_image(self, image):
        """Preprocess the image for detection."""
        pass

    @abstractmethod
    def postprocess_mask(self, mask):
        """Postprocess the mask obtained from detection."""
        pass