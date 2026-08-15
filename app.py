import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import json
from werkzeug.utils import secure_filename
import timm
import cv2
import base64
import io

# =========================
# 🔥 Grad-CAM Class
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().detach().numpy()

        # Resize to input size
        cam = cv2.resize(cam, (224, 224))

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def __call__(self, input_tensor, targets=None):
        """Compatibility method to work with existing code"""
        class_idx = targets[0].category_idx if targets else None
        return [self.generate(input_tensor, class_idx)]

app = Flask(__name__, static_folder='Frontend', template_folder='Frontend')

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
IMG_SIZE = 224

# Model Classes (matching your training code)
class AttentionFusion(nn.Module):
    def __init__(self, cnn_dim, vit_dim):
        super().__init__()
        self.cnn_proj = nn.Linear(cnn_dim, 512)
        self.vit_proj = nn.Linear(vit_dim, 512)
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, cnn_feat, vit_feat):
        cnn_feat = self.cnn_proj(cnn_feat)
        vit_feat = self.vit_proj(vit_feat)
        concat = torch.cat((cnn_feat, vit_feat), dim=1)
        weights = self.attention(concat)
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        fused = w1 * cnn_feat + w2 * vit_feat
        return fused

class AlzheimerHybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # EfficientNet B0
        self.cnn = timm.create_model(
            "efficientnet_b0", 
            pretrained=True, 
            num_classes=0
        )
        # Swin Transformer
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224", 
            pretrained=True, 
            num_classes=0
        )
        
        cnn_dim = self.cnn.num_features
        vit_dim = self.swin.num_features
        
        self.fusion = AttentionFusion(cnn_dim, vit_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.swin(x)
        fused = self.fusion(cnn_feat, vit_feat)
        out = self.classifier(fused)
        return out

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerHybridModel().to(device)

# Load the trained model
model_path = os.path.join('Backend', 'best_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"Model file not found at {model_path}")

# Class labels (adjust according to your dataset)
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Image transformations (matching your training transforms)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]
    )
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_brain_region(cam):
    """Detect which brain region received most attention"""
    h, w = cam.shape
    
    center = cam[h//4:3*h//4, w//4:3*w//4].mean()
    left = cam[:, :w//2].mean()
    right = cam[:, w//2:].mean()
    top = cam[:h//2, :].mean()
    bottom = cam[h//2:, :].mean()
    
    region_scores = {
        "center (ventricles / brain atrophy)": center,
        "left hemisphere (cortical region)": left,
        "right hemisphere (cortical region)": right,
        "frontal lobe (top region)": top,
        "cerebellum (bottom region)": bottom
    }
    
    return max(region_scores, key=region_scores.get)

def generate_explanation(pred_class, confidence, cam_intensity, region):
    """Generate medical explanation based on prediction and attention"""
    # Confidence level
    if confidence > 0.9:
        conf_text = "very high confidence"
    elif confidence > 0.75:
        conf_text = "good confidence"
    else:
        conf_text = "low confidence"
    
    # Activation strength
    if cam_intensity > 0.6:
        focus = "strong activation"
    elif cam_intensity > 0.3:
        focus = "moderate activation"
    else:
        focus = "weak activation"
    
    # Medical explanation
    if pred_class == "Moderate Demented":
        return f"Predicted Moderate Demented with {conf_text}. The model shows {focus} in {region}, indicating significant brain atrophy and structural degeneration."
    
    elif pred_class == "Mild Demented":
        return f"Predicted Mild Demented with {conf_text}. The model shows {focus} in {region}, suggesting early-stage cortical changes."
    
    elif pred_class == "Very Mild Demented":
        return f"Predicted Very Mild Demented with {conf_text}. The model shows {focus} in {region}, indicating subtle structural variations."
    
    elif pred_class == "Non Demented":
        return f"Predicted Non Demented with {conf_text}. The model shows {focus} in {region}, with no significant abnormality detected."
    
    else:
        return "No explanation available."

def generate_gradcam(image, model, class_idx):
    """Generate Grad-CAM heatmap and overlay matching Colab reference exactly"""
    try:
        # Enable gradients for Grad-CAM
        for param in model.parameters():
            param.requires_grad = True
        
        # Convert PIL to numpy array and ensure correct size
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert RGB to BGR (simulate cv2.imread behavior from reference)
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        
        # Transform for model (exact match to reference)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB for transform (as reference does)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(next(model.parameters()).device)
        
        # Setup Grad-CAM
        target_layer = model.cnn.conv_head
        gradcam = GradCAM(model, target_layer)
        
        # Generate CAM
        cam = gradcam.generate(tensor, class_idx)
        
        # Create heatmap exactly like reference
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )
        
        # Create overlay exactly like reference (using RGB for both)
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64 for web display
        _, overlay_buffer = cv2.imencode('.png', overlay)
        overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
        
        # Original image for display
        _, orig_buffer = cv2.imencode('.png', img_rgb)
        original_base64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        return {
            'overlay': overlay_base64,
            'original': original_base64,
            'heatmap_intensity': float(np.mean(cam)),
            'region': detect_brain_region(cam)
        }
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def preprocess_image(image):
    """Preprocess PIL Image for model prediction"""
    image = image.convert('RGB')
    image = test_transform(image).unsqueeze(0)
    return image.to(device)

def predict_alzheimer(image_file):
    """Make prediction on uploaded image file with Grad-CAM visualization"""
    try:
        # Open image directly from file storage
        image = Image.open(image_file)
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get all class probabilities
            all_probs = probabilities.cpu().numpy()[0]
            pred_idx = predicted.item()
            pred_class = class_labels[pred_idx]
            
        # Generate Grad-CAM visualization
        gradcam_result = generate_gradcam(image, model, pred_idx)
        
        # Generate explanation
        if gradcam_result:
            explanation = generate_explanation(
                pred_class,
                float(confidence.item()),
                gradcam_result['heatmap_intensity'],
                gradcam_result['region']
            )
        else:
            explanation = f"Predicted {pred_class} with {float(confidence.item())*100:.1f}% confidence."
        
        result = {
            'prediction': pred_class,
            'confidence': float(confidence.item()),
            'class_probabilities': {
                class_labels[i]: float(all_probs[i]) 
                for i in range(len(class_labels))
            },
            'explanation': explanation
        }
        
        # Add Grad-CAM results if available
        if gradcam_result:
            result['images'] = {
                'original': gradcam_result['original'],
                'overlay': gradcam_result['overlay']
            }
            result['attention_region'] = gradcam_result['region']
            result['heatmap_intensity'] = gradcam_result['heatmap_intensity']
        
        return result
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('Frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, images)"""
    return send_from_directory('Frontend', filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Make prediction directly from file storage
            result = predict_alzheimer(file)
            
            if 'error' in result:
                return jsonify(result), 500
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'model_name': 'Alzheimer Hybrid CNN-ViT Model',
        'architecture': 'EfficientNet-B0 + Swin Transformer with Attention Fusion',
        'classes': class_labels,
        'input_size': f'{IMG_SIZE}x{IMG_SIZE}',
        'device': str(device)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists(model_path)
    })

if __name__ == '__main__':
    print("Starting Alzheimer's Prediction Web Application...")
    print(f"Model file: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")
    print("Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
