from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pytesseract
import cv2
import numpy as np
# import os
# import string


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


@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image uploaded'}, status=400)
        image_file = request.FILES['image']
        try:
            image_np = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            # text = SimpleOCR.image_to_string(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            gray = cv2.medianBlur(gray, 3)      
                  
            text = pytesseract.image_to_string(gray)
            return JsonResponse({
                'success': True,
                'text': text.strip()
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

    