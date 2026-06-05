import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    """
    
    def __init__(self, model, class_names):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        
    def make_gradcam_heatmap(self, img_array, pred_index=None, last_conv_layer_name=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_array: Preprocessed image array
            pred_index: Index of the class to generate heatmap for (if None, uses predicted class)
            last_conv_layer_name: Name of the last convolutional layer
        
        Returns:
            Heatmap array
        """
        
        # If no specific layer is provided, find the last conv layer
        if last_conv_layer_name is None:
            last_conv_layer_name = self._find_last_conv_layer()
        
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = Model(
            [self.model.inputs], 
            [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )
        
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def _find_last_conv_layer(self):
        """
        Find the last convolutional layer in the model.
        
        Returns:
            Name of the last convolutional layer
        """
        for layer in reversed(self.model.layers):
            # Check if layer has 4D output (conv layer)
            if len(layer.output_shape) == 4:
                return layer.name
        
        # If no conv layer found, try to find in base model
        if hasattr(self.model.layers[0], 'layers'):
            for layer in reversed(self.model.layers[0].layers):
                if len(layer.output_shape) == 4:
                    return layer.name
        
        raise ValueError("Could not find a 4D layer (convolutional layer)")
    
    def generate_gradcam_visualization(self, img_path, save_path=None, alpha=0.4):
        """
        Generate and save Grad-CAM visualization.
        
        Args:
            img_path: Path to the input image
            save_path: Path to save the visualization (if None, returns the image)
            alpha: Transparency of the heatmap overlay
        
        Returns:
            Combined visualization image
        """
        
        # Preprocess image
        from .efficientnet_model import preprocess_image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        preds = self.model.predict(img_array)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]
        
        # Generate heatmap
        heatmap = self.make_gradcam_heatmap(img_array, pred_index=pred_class)
        
        # Load original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((224, 224))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
        axes[1].axis('off')
        
        # Superimposed image
        axes[2].imshow(superimposed_img)
        axes[2].set_title(f'Prediction: {self.class_names[pred_class]}\nConfidence: {confidence:.2%}', 
                         fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return fig
    
    def generate_class_specific_heatmaps(self, img_path, save_dir=None):
        """
        Generate Grad-CAM heatmaps for all classes.
        
        Args:
            img_path: Path to the input image
            save_dir: Directory to save visualizations
        
        Returns:
            List of generated visualizations
        """
        
        from .efficientnet_model import preprocess_image
        import os
        
        img_array = preprocess_image(img_path)
        preds = self.model.predict(img_array)
        
        visualizations = []
        
        for class_idx in range(len(self.class_names)):
            # Generate heatmap for this class
            heatmap = self.make_gradcam_heatmap(img_array, pred_index=class_idx)
            
            # Load and process original image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Original image
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Heatmap
            im = axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title(f'{self.class_names[class_idx]}\nScore: {preds[0][class_idx]:.3f}')
            axes[1].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            if save_dir:
                save_path = os.path.join(save_dir, f'gradcam_{self.class_names[class_idx].replace(" ", "_")}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualizations.append(save_path)
            else:
                visualizations.append(fig)
        
        return visualizations