import cv2
import numpy as np
import pytesseract
import re
from typing import Union

class OCRManager:
    def __init__(self):
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        self.tesseract_config = '--psm 6'
        self.kernel = np.ones((2, 2), np.uint8)

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def threshold(self, image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
        return binary

    def noise_removal(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def thick_font(self, image: np.ndarray) -> np.ndarray:
        image = cv2.bitwise_not(image)
        image = cv2.dilate(image, self.kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def correct_bh_codes(self, text: str) -> str:
        return re.sub(r'BH-\d+@?', 'BH-2000', text)

    def ocr(self, image_input: Union[bytes, np.ndarray]) -> str:
        """
        Main OCR method that handles both:
        - Raw image bytes (from base64 decoded input)
        - Already decoded numpy arrays
        
        Args:
            image_input: Either raw image bytes or decoded numpy array
            
        Returns:
            Extracted text from image
        """
        # Convert bytes to numpy array if needed
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from bytes")
        else:
            image = image_input

        # Process image through OCR pipeline
        gray = self.grayscale(image)
        thresh = self.threshold(gray)
        no_noise = self.noise_removal(thresh)
        thickened = self.thick_font(no_noise)
        text = pytesseract.image_to_string(thickened, config=self.tesseract_config)
        return self.correct_bh_codes(text)

    # Keep ocr_from_bytes for backward compatibility
    def ocr_from_bytes(self, image_bytes: bytes) -> str:
        """Alternative method that explicitly takes bytes"""
        return self.ocr(image_bytes)
