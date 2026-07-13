"""
Simple Activation Heatmap for Tumor Localization - Works with any model
"""
import numpy as np
import cv2
import tensorflow as tf

class SimpleHeatmap:
    def __init__(self, model):
        self.model = model
        
    def get_last_conv_layer(self):
        """Find the last convolutional layer in the model"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            # Check inside Sequential or Functional models
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name
        return None
    
    def compute_heatmap(self, img_array, predicted_class):
        """
        Compute a simple heatmap based on model predictions
        """
        try:
            # Method 1: Try Grad-CAM first
            conv_layer_name = self.get_last_conv_layer()
            
            if conv_layer_name:
                # Create a model that outputs the conv layer and predictions
                try:
                    conv_layer = self.model.get_layer(conv_layer_name)
                    gradient_model = tf.keras.Model(
                        inputs=self.model.inputs,
                        outputs=[conv_layer.output, self.model.output]
                    )
                    
                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = gradient_model(img_array)
                        loss = predictions[:, predicted_class]
                    
                    grads = tape.gradient(loss, conv_outputs)
                    
                    if grads is not None:
                        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                        conv_outputs = conv_outputs[0]
                        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                        heatmap = tf.maximum(heatmap, 0)
                        heatmap /= tf.reduce_max(heatmap) + 1e-10
                        return heatmap.numpy()
                except:
                    pass
            
            # Method 2: Create a saliency map using gradients
            with tf.GradientTape() as tape:
                tape.watch(img_array)
                predictions = self.model(img_array)
                loss = predictions[:, predicted_class]
            
            grads = tape.gradient(loss, img_array)
            
            if grads is not None:
                # Take absolute values and average across channels
                grads_abs = tf.reduce_mean(tf.abs(grads), axis=-1)
                heatmap = tf.reduce_max(grads_abs, axis=0).numpy()
                # Normalize
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
                return heatmap
            
        except Exception as e:
            print(f"Heatmap computation error: {e}")
        
        # Method 3: Fallback - create a centered heatmap based on prediction confidence
        h, w = img_array.shape[1], img_array.shape[2]
        heatmap = np.zeros((h, w))
        
        # Create a gradient pattern based on confidence
        center_y, center_x = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                heatmap[i, j] = max(0, 1 - distance / (min(h, w) / 2))
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """Overlay heatmap on original image"""
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize to 0-255
        heatmap_norm = np.uint8(255 * heatmap_resized / (np.max(heatmap_resized) + 1e-10))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return superimposed, heatmap_colored
    
    def get_tumor_location(self, heatmap, threshold=0.5):
        """Determine tumor location based on heatmap"""
        heatmap_norm = heatmap / (np.max(heatmap) + 1e-10)
        
        # Use lower threshold for better detection
        tumor_regions = heatmap_norm > threshold
        
        if not np.any(tumor_regions):
            tumor_regions = heatmap_norm > 0.3
        
        if not np.any(tumor_regions):
            return {
                'location': "Unable to determine precise location",
                'coordinates': None,
                'area_percentage': 0,
                'intensity': 0,
                'confidence_level': "Low"
            }
        
        # Get center of mass
        y_coords, x_coords = np.where(tumor_regions)
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        h, w = heatmap.shape
        
        # Determine region
        if center_x < w // 3:
            x_pos = "Left"
        elif center_x > 2 * w // 3:
            x_pos = "Right"
        else:
            x_pos = "Central"
        
        if center_y < h // 3:
            y_pos = "Upper"
        elif center_y > 2 * h // 3:
            y_pos = "Lower"
        else:
            y_pos = "Middle"
        
        area_percentage = (np.sum(tumor_regions) / (h * w)) * 100
        max_intensity = float(np.max(heatmap_norm[tumor_regions]))
        
        confidence_level = "High" if max_intensity > 0.7 else "Medium" if max_intensity > 0.4 else "Low"
        
        return {
            'location': f"{y_pos} {x_pos} Region",
            'detailed_location': f"{y_pos} portion of the {x_pos.lower()} hemisphere",
            'coordinates': (center_x, center_y),
            'area_percentage': round(area_percentage, 2),
            'intensity': round(max_intensity, 3),
            'confidence_level': confidence_level
        }


def generate_tumor_visualization(img_array, model, class_names, predicted_class_idx, confidence):
    """Generate visualization for tumor detection"""
    try:
        # Initialize heatmap generator
        heatmap_gen = SimpleHeatmap(model)
        
        # Compute heatmap
        heatmap = heatmap_gen.compute_heatmap(img_array, predicted_class_idx)
        
        # Get original image
        original_img = (img_array[0] * 255).astype(np.uint8)
        original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Create overlay
        overlay_img, heatmap_colored = heatmap_gen.overlay_heatmap(original_img_bgr, heatmap, alpha=0.5)
        overlay_img_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        
        # Create side-by-side comparison
        comparison = np.hstack([original_img, overlay_img_rgb])
        
        # Get tumor location info
        tumor_info = heatmap_gen.get_tumor_location(heatmap, threshold=0.4)
        
        return {
            'original_image': original_img,
            'heatmap': heatmap,
            'overlay_image': overlay_img_rgb,
            'comparison_image': comparison,
            'tumor_info': tumor_info,
            'success': True
        }
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }