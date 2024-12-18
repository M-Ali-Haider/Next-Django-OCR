import cv2
import numpy as np
import string
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pytesseract

# class CharacterTemplateGenerator:
#     @staticmethod
#     def create_character_template(char, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2, image_size=(50, 50)):
#         image = np.zeros(image_size, dtype=np.uint8)
#         image.fill(255)
#         (text_width, text_height), _ = cv2.getTextSize(
#             char, font, font_scale, thickness
#         )        
#         x = (image_size[1] - text_width) // 2
#         y = (image_size[0] + text_height) // 2
#         cv2.putText(
#             image, 
#             char, 
#             (x, y), 
#             font, 
#             font_scale, 
#             (0, 0, 0),  
#             thickness
#         )        
#         _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
#         return binary
    
#     @classmethod
#     def generate_all_character_templates(cls, output_dir='character_templates'):
#         os.makedirs(output_dir, exist_ok=True)
#         characters = string.ascii_letters + string.digits
#         templates = {}
#         for char in characters:
#             template = cls.create_character_template(char)
#             cv2.imwrite(os.path.join(output_dir, f'{char}_template.png'), template)
#             templates[char] = template
#         return templates
    
#     @staticmethod
#     def load_templates_from_directory(directory='character_templates'):
#         templates = {}
#         if not os.path.exists(directory):
#             raise FileNotFoundError(f"Directory {directory} does not exist")
        
#         for filename in os.listdir(directory):
#             if filename.endswith('_template.png'):
#                 char = filename.split('_')[0]
#                 template = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
#                 _, binary = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
#                 templates[char] = binary
        
#         return templates


# generator = CharacterTemplateGenerator()
# templates = generator.generate_all_character_templates() 
# CHARACTER_TEMPLATES = templates


# class SimpleOCR:
#     @staticmethod
#     def preprocess_image(image):
#         if len(image.shape) > 2:
#             gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image
        
#         _, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         binary = cv2.medianBlur(binary,3)
#         return binary
    
#     @staticmethod
#     def find_characters(binary_image):
#         contours,_ = cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#         char_contours = [
#             cv2.boundingRect(cnt) for cnt in contours
#             if 10 < cv2.contourArea(cnt) < 500
#         ]
#         return sorted(char_contours,key=lambda x:x[0])
    
#     @classmethod
#     def match_character(cls,char_image):
#         best_match = None
#         best_score = float('inf')
#         char_image = cv2.resize(char_image,(20,20))
#         for char,template in CHARACTER_TEMPLATES.items():
#             template_resized = cv2.resize(template,(20,20))
#             result = cv2.matchTemplate(char_image,template_resized,cv2.TM_SQDIFF_NORMED)
#             score = np.min(result)

#             if score < best_score:
#                 best_score = score
#                 best_match = char
#         return best_match or '?'

#     @classmethod
#     def image_to_string(cls,image):
#         binary_image = cls.preprocess_image(image)
#         characters = cls.find_characters(binary_image)
#         recognized_text = []
#         for x,y,w,h in characters:
#             char_image = binary_image[y:y+h,x:x+w]
#             recognized_char = cls.match_character(char_image)
#             recognized_text.append(recognized_char)
        
#         return ''.join(recognized_text)


# @csrf_exempt
# def process_image(request):
#     if request.method == 'POST':
#         if 'image' not in request.FILES:
#             return JsonResponse({'error': 'No image uploaded'}, status=400)
#         image_file = request.FILES['image']
#         try:
#             image_np = np.frombuffer(image_file.read(), np.uint8)
#             image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
#             # text = SimpleOCR.image_to_string(image)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#             gray = cv2.medianBlur(gray, 3)      
                  
#             text = pytesseract.image_to_string(gray)
#             return JsonResponse({
#                 'success': True,
#                 'text': text.strip()
#             })
#         except Exception as e:
#             return JsonResponse({
#                 'success': False,
#                 'error': str(e)
#             }, status=500)
#     return JsonResponse({'error': 'Invalid request method'}, status=405)

    
    
    


class ImprovedOCR:
    def __init__(self, model_path='ocr_model.h5'):
        self.model_path = model_path
        self.model = self.build_or_load_model()
        
    def build_or_load_model(self):
        """Build or load a CNN model for character recognition"""
        if os.path.exists(self.model_path):
            return tf.keras.models.load_model(self.model_path)
        
        # Create a new CNN model if no saved model exists
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(36, activation='softmax')  # 26 letters + 10 digits
        ])
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def preprocess_image(self, image):
        """Advanced image preprocessing"""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Adaptive thresholding for better binarization
        binary = cv2.adaptiveThreshold(
            gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )
        
        # Noise removal
        binary = cv2.medianBlur(binary, 3)
        
        return binary
    
    def segment_characters(self, binary_image):
        """Improved character segmentation"""
        # Find contours with more robust filtering
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and sort contours
        char_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # More sophisticated filtering
            aspect_ratio = w / float(h)
            if (20 < area < 1000 and 
                0.2 < aspect_ratio < 2.0 and 
                w > 5 and h > 5):
                char_contours.append((x, y, w, h))
        
        # Sort left to right
        return sorted(char_contours, key=lambda x: x[0])
    
    def prepare_character_for_recognition(self, binary_image, bbox):
        """Prepare a character image for the neural network"""
        x, y, w, h = bbox
        char_img = binary_image[y:y+h, x:x+w]
        
        # Resize and pad to match model input
        char_img = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize
        char_img = char_img / 255.0
        
        # Reshape for model input
        char_img = char_img.reshape((28, 28, 1))
        
        return char_img
    
    def train_model(self, training_images, labels):
        """Train the OCR model"""
        # Assuming training_images is a list of preprocessed character images
        # and labels is a corresponding list of character labels
        
        # Convert images and labels to numpy arrays
        X = np.array(training_images)
        y = self.encode_labels(labels)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train, 
            epochs=10, 
            validation_data=(X_val, y_val)
        )
        
        # Save the model
        self.model.save(self.model_path)
        return history
    
    def encode_labels(self, labels):
        """Convert text labels to one-hot encoded vectors"""
        # Create mapping of characters to indices
        char_indices = {
            **{char: i for i, char in enumerate(string.ascii_uppercase)},
            **{str(digit): i+26 for digit in range(10)}
        }
        
        # One-hot encode
        encoded = np.zeros((len(labels), 36))
        for i, label in enumerate(labels):
            encoded[i, char_indices[label.upper()]] = 1
        
        return encoded
    
    def decode_prediction(self, prediction):
        """Convert model prediction back to character"""
        # Mapping of indices back to characters
        char_map = (
            list(string.ascii_uppercase) + 
            list(map(str, range(10)))
        )
        return char_map[np.argmax(prediction)]
    
    def recognize_text(self, image):
        """Main method to recognize text in an image"""
        # Preprocess the entire image
        binary_image = self.preprocess_image(image)
        
        # Segment characters
        character_bboxes = self.segment_characters(binary_image)
        
        # Recognize each character
        recognized_text = []
        for bbox in character_bboxes:
            # Prepare character for recognition
            char_img = self.prepare_character_for_recognition(binary_image, bbox)
            
            # Predict
            prediction = self.model.predict(np.array([char_img]))
            recognized_char = self.decode_prediction(prediction[0])
            
            recognized_text.append(recognized_char)
        
        return ''.join(recognized_text)

# Example usage
if __name__ == '__main__':
    # Initialize OCR
    ocr = ImprovedOCR()
    
    # Load and recognize text from an image
    image = cv2.imread('sample_text.png', cv2.IMREAD_GRAYSCALE)
    recognized_text = ocr.recognize_text(image)
    print("Recognized Text:", recognized_text)