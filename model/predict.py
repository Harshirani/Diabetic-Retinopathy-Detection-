
# """
# predict.py - Diabetic Retinopathy Detection Prediction Script
# """

# import tensorflow as tf
# import numpy as np
# import cv2
# import os
# import matplotlib.pyplot as plt
# from pathlib import Path

# class DiabetinRetinopathyPredictor:
#     def __init__(self, model_path='diabetic_retinopathy_model.h5'):
#         """
#         Initialize the predictor with a trained model.
        
#         Args:
#             model_path: Path to the saved model file
#         """
#         self.model_path = model_path
#         self.model = None
#         self.class_names = [
#             'No DR',
#             'Mild DR', 
#             'Moderate DR',
#             'Severe DR',
#             'Proliferative DR'
#         ]
#         self.img_size = (224, 224)
        
#         self.load_model()
    
#     def load_model(self):
#         """Load the trained model"""
#         try:
#             if not os.path.exists(self.model_path):
#                 raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
#             self.model = tf.keras.models.load_model(self.model_path)
#             print(f"Model loaded successfully from {self.model_path}")
            
#             # Print model summary
#             print(f"Input shape: {self.model.input_shape}")
#             print(f"Output shape: {self.model.output_shape}")
#             print(f"Number of classes: {len(self.class_names)}")
            
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             self.model = None
    
#     def preprocess_image(self, image_path):
#         """
#         Preprocess image for prediction.
        
#         Args:
#             image_path: Path to the image file
            
#         Returns:
#             Preprocessed image array
#         """
#         try:
#             # Load image
#             image = tf.keras.preprocessing.image.load_img(
#                 image_path, 
#                 target_size=self.img_size
#             )
            
#             # Convert to array
#             img_array = tf.keras.preprocessing.image.img_to_array(image)
#             img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
            
#             # Normalize pixel values to [0,1]
#             img_array = img_array / 255.0
            
#             return img_array
            
#         except Exception as e:
#             print(f"Error preprocessing image: {e}")
#             return None
    
#     def predict_single_image(self, image_path, show_confidence=True):
#         """
#         Predict diabetic retinopathy grade for a single image.
        
#         Args:
#             image_path: Path to the image file
#             show_confidence: Whether to show confidence scores for all classes
            
#         Returns:
#             Dictionary with prediction results
#         """
#         if self.model is None:
#             print("No model loaded!")
#             return None
        
#         try:
#             # Preprocess image
#             img_array = self.preprocess_image(image_path)
#             if img_array is None:
#                 return None
            
#             # Make prediction
#             predictions = self.model.predict(img_array, verbose=0)
#             predicted_class_idx = np.argmax(predictions[0])
#             confidence = predictions[0][predicted_class_idx]
#             predicted_class = self.class_names[predicted_class_idx]
            
#             # Prepare results
#             results = {
#                 'image_path': image_path,
#                 'predicted_class': predicted_class,
#                 'predicted_class_index': predicted_class_idx,
#                 'confidence': float(confidence),
#                 'all_probabilities': {
#                     self.class_names[i]: float(predictions[0][i]) 
#                     for i in range(len(self.class_names))
#                 }
#             }
            
#             # Display results
#             print(f"\nPrediction for: {os.path.basename(image_path)}")
#             print(f"Predicted Class: {predicted_class}")
#             print(f"Confidence: {confidence:.2%}")
            
#             if show_confidence:
#                 print("\nAll class probabilities:")
#                 for i, class_name in enumerate(self.class_names):
#                     prob = predictions[0][i]
#                     print(f"  {class_name}: {prob:.3f} ({prob:.1%})")
            
#             return results
            
#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             return None
    
#     def predict_batch(self, image_folder, save_results=True):
#         """
#         Predict diabetic retinopathy for all images in a folder.
        
#         Args:
#             image_folder: Path to folder containing images
#             save_results: Whether to save results to CSV
            
#         Returns:
#             List of prediction results
#         """
#         if self.model is None:
#             print("No model loaded!")
#             return None
        
#         try:
#             # Get all image files
#             image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
#             image_files = []
            
#             for ext in image_extensions:
#                 image_files.extend(Path(image_folder).glob(f'*{ext}'))
#                 image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
            
#             if not image_files:
#                 print(f"No images found in {image_folder}")
#                 return []
            
#             print(f"Found {len(image_files)} images to process...")
            
#             all_results = []
            
#             for i, image_path in enumerate(image_files, 1):
#                 print(f"\nProcessing {i}/{len(image_files)}: {image_path.name}")
                
#                 result = self.predict_single_image(str(image_path), show_confidence=False)
#                 if result:
#                     all_results.append(result)
            
#             # Save results to CSV if requested
#             if save_results and all_results:
#                 self.save_results_to_csv(all_results, image_folder)
            
#             return all_results
            
#         except Exception as e:
#             print(f"Error during batch prediction: {e}")
#             return []
    
#     def save_results_to_csv(self, results, output_folder):
#         """Save prediction results to CSV file"""
#         try:
#             import pandas as pd
            
#             # Prepare data for CSV
#             csv_data = []
#             for result in results:
#                 row = {
#                     'image_path': result['image_path'],
#                     'image_name': os.path.basename(result['image_path']),
#                     'predicted_class': result['predicted_class'],
#                     'confidence': result['confidence']
#                 }
                
#                 # Add individual class probabilities
#                 for class_name, prob in result['all_probabilities'].items():
#                     row[f'{class_name}_probability'] = prob
                
#                 csv_data.append(row)
            
#             # Create DataFrame and save
#             df = pd.DataFrame(csv_data)
#             csv_path = os.path.join(output_folder, 'prediction_results.csv')
#             df.to_csv(csv_path, index=False)
            
#             print(f"\nResults saved to: {csv_path}")
            
#         except ImportError:
#             print("pandas not installed. Install with: pip install pandas")
#         except Exception as e:
#             print(f"Error saving results: {e}")
    
#     def visualize_prediction(self, image_path, save_path=None):
#         """
#         Visualize prediction with the original image and probability bar chart.
        
#         Args:
#             image_path: Path to the image file
#             save_path: Path to save the visualization (optional)
#         """
#         if self.model is None:
#             print("No model loaded!")
#             return
        
#         try:
#             # Get prediction
#             result = self.predict_single_image(image_path, show_confidence=False)
#             if result is None:
#                 return
            
#             # Load original image for display
#             original_image = cv2.imread(image_path)
#             original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
#             # Create figure with subplots
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
#             # Display original image
#             ax1.imshow(original_image)
#             ax1.set_title(f'Original Image\n{os.path.basename(image_path)}', fontsize=14)
#             ax1.axis('off')
            
#             # Display probability bar chart
#             classes = list(result['all_probabilities'].keys())
#             probabilities = list(result['all_probabilities'].values())
            
#             bars = ax2.barh(classes, probabilities)
#             ax2.set_xlabel('Probability')
#             ax2.set_title(f'Prediction: {result["predicted_class"]}\nConfidence: {result["confidence"]:.2%}', 
#                          fontsize=14)
#             ax2.set_xlim(0, 1)
            
#             # Color the predicted class bar differently
#             predicted_idx = result['predicted_class_index']
#             bars[predicted_idx].set_color('red')
            
#             # Add probability values on bars
#             for i, (bar, prob) in enumerate(zip(bars, probabilities)):
#                 width = bar.get_width()
#                 ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
#                         f'{prob:.3f}', ha='left', va='center')
            
#             plt.tight_layout()
            
#             if save_path:
#                 plt.savefig(save_path, dpi=150, bbox_inches='tight')
#                 print(f"Visualization saved to: {save_path}")
#             else:
#                 plt.show()
            
#         except Exception as e:
#             print(f"Error creating visualization: {e}")


# def main():
#     """Main function to run predictions"""
#     print("Diabetic Retinopathy Detection - Prediction Tool")
#     print("=" * 50)
    
#     # Initialize predictor
#     predictor = DiabetinRetinopathyPredictor()
    
#     if predictor.model is None:
#         print("Please train a model first using train_model.py")
#         return
    
#     while True:
#         print("\nChoose an option:")
#         print("1. Predict single image")
#         print("2. Predict batch of images")
#         print("3. Predict with visualization")
#         print("4. Exit")
        
#         choice = input("Enter your choice (1-4): ").strip()
        
#         if choice == '1':
#             image_path = input("Enter path to image file: ").strip()
#             if os.path.exists(image_path):
#                 result = predictor.predict_single_image(image_path)
#             else:
#                 print("Image file not found!")
        
#         elif choice == '2':
#             folder_path = input("Enter path to folder containing images: ").strip()
#             if os.path.exists(folder_path):
#                 results = predictor.predict_batch(folder_path)
#                 print(f"\nProcessed {len(results)} images successfully!")
#             else:
#                 print("Folder not found!")
        
#         elif choice == '3':
#             image_path = input("Enter path to image file: ").strip()
#             if os.path.exists(image_path):
#                 save_viz = input("Save visualization? (y/n): ").strip().lower()
#                 if save_viz == 'y':
#                     save_path = f"prediction_viz_{os.path.basename(image_path)}.png"
#                     predictor.visualize_prediction(image_path, save_path)
#                 else:
#                     predictor.visualize_prediction(image_path)
#             else:
#                 print("Image file not found!")
        
#         elif choice == '4':
#             print("Goodbye!")
#             break
        
#         else:
#             print("Invalid choice!")


# if __name__ == "__main__":
#     main()






"""
predict.py - Diabetic Retinopathy Detection Prediction Script
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys

class DiabetinRetinopathyPredictor:
    def __init__(self, model_path='diabetic_retinopathy_model.h5'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = [
            'No DR',
            'Mild DR', 
            'Moderate DR',
            'Severe DR',
            'Proliferative DR'
        ]
        self.img_size = (224, 224)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Print model summary
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
            print(f"Number of classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=self.img_size
            )
            
            # Convert to array
            img_array = tf.keras.preprocessing.image.img_to_array(image)
            img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
            
            # Normalize pixel values to [0,1]
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_single_image(self, image_path, show_confidence=True):
        """
        Predict diabetic retinopathy grade for a single image.
        
        Args:
            image_path: Path to the image file
            show_confidence: Whether to show confidence scores for all classes
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            print("No model loaded!")
            return None
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            if img_array is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            # Prepare results
            results = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'predicted_class_index': predicted_class_idx,
                'confidence': float(confidence),
                'all_probabilities': {
                    self.class_names[i]: float(predictions[0][i]) 
                    for i in range(len(self.class_names))
                }
            }
            
            # Display results
            print(f"\nPrediction for: {os.path.basename(image_path)}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            
            if show_confidence:
                print("\nAll class probabilities:")
                for i, class_name in enumerate(self.class_names):
                    prob = predictions[0][i]
                    print(f"  {class_name}: {prob:.3f} ({prob:.1%})")
            
            return results
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def predict_batch(self, image_folder, save_results=True):
        """
        Predict diabetic retinopathy for all images in a folder.
        
        Args:
            image_folder: Path to folder containing images
            save_results: Whether to save results to CSV
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            print("No model loaded!")
            return None
        
        try:
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(image_folder).glob(f'*{ext}'))
                image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"No images found in {image_folder}")
                return []
            
            print(f"Found {len(image_files)} images to process...")
            
            all_results = []
            
            for i, image_path in enumerate(image_files, 1):
                print(f"\nProcessing {i}/{len(image_files)}: {image_path.name}")
                
                result = self.predict_single_image(str(image_path), show_confidence=False)
                if result:
                    all_results.append(result)
            
            # Save results to CSV if requested
            if save_results and all_results:
                self.save_results_to_csv(all_results, image_folder)
            
            return all_results
            
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return []
    
    def save_results_to_csv(self, results, output_folder):
        """Save prediction results to CSV file"""
        try:
            import pandas as pd
            
            # Prepare data for CSV
            csv_data = []
            for result in results:
                row = {
                    'image_path': result['image_path'],
                    'image_name': os.path.basename(result['image_path']),
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence']
                }
                
                # Add individual class probabilities
                for class_name, prob in result['all_probabilities'].items():
                    row[f'{class_name}_probability'] = prob
                
                csv_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(output_folder, 'prediction_results.csv')
            df.to_csv(csv_path, index=False)
            
            print(f"\nResults saved to: {csv_path}")
            
        except ImportError:
            print("pandas not installed. Install with: pip install pandas")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with the original image and probability bar chart.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the visualization (optional)
        """
        if self.model is None:
            print("No model loaded!")
            return
        
        try:
            # Get prediction
            result = self.predict_single_image(image_path, show_confidence=False)
            if result is None:
                return
            
            # Load original image for display
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display original image
            ax1.imshow(original_image)
            ax1.set_title(f'Original Image\n{os.path.basename(image_path)}', fontsize=14)
            ax1.axis('off')
            
            # Display probability bar chart
            classes = list(result['all_probabilities'].keys())
            probabilities = list(result['all_probabilities'].values())
            
            bars = ax2.barh(classes, probabilities)
            ax2.set_xlabel('Probability')
            ax2.set_title(f'Prediction: {result["predicted_class"]}\nConfidence: {result["confidence"]:.2%}', 
                         fontsize=14)
            ax2.set_xlim(0, 1)
            
            # Color the predicted class bar differently
            predicted_idx = result['predicted_class_index']
            bars[predicted_idx].set_color('red')
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")


def get_user_input(prompt):
    """Get user input with better error handling"""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Input error: {e}")
        return ""


def main():
    """Main function to run predictions"""
    print("Diabetic Retinopathy Detection - Prediction Tool")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DiabetinRetinopathyPredictor()
    
    if predictor.model is None:
        print("Please train a model first using train_model.py")
        print("Make sure the model file 'diabetic_retinopathy_model.h5' exists in the current directory.")
        return
    
    while True:
        try:
            print("\n" + "="*50)
            print("Choose an option:")
            print("1. Predict single image")
            print("2. Predict batch of images")
            print("3. Predict with visualization")
            print("4. Exit")
            print("="*50)
            
            choice = get_user_input("Enter your choice (1-4): ")
            
            # Handle empty input or invalid characters
            if not choice:
                print("Please enter a valid choice (1-4).")
                continue
            
            # Clean the choice (remove any extra characters)
            choice = choice[0] if choice else ""
            
            if choice == '1':
                print("\n--- Single Image Prediction ---")
                image_path = get_user_input("Enter path to image file: ")
                
                if not image_path:
                    print("No path provided.")
                    continue
                
                # Handle relative paths based on your screenshot structure
                if not os.path.isabs(image_path) and not os.path.exists(image_path):
                    # Try common relative paths
                    possible_paths = [
                        image_path,
                        os.path.join("data", image_path),
                        os.path.join("data", "val", "Moderate", image_path),
                        os.path.join("data", "val", "Mild", image_path),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                
                if os.path.exists(image_path):
                    result = predictor.predict_single_image(image_path)
                    if result:
                        print(f"\n✅ Prediction completed successfully!")
                else:
                    print(f"❌ Image file not found: {image_path}")
                    print("Please check the path and try again.")
            
            elif choice == '2':
                print("\n--- Batch Prediction ---")
                folder_path = get_user_input("Enter path to folder containing images: ")
                
                if not folder_path:
                    print("No path provided.")
                    continue
                
                # Handle relative paths
                if not os.path.isabs(folder_path) and not os.path.exists(folder_path):
                    possible_paths = [
                        folder_path,
                        os.path.join("data", folder_path),
                        os.path.join("data", "val", folder_path),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            folder_path = path
                            break
                
                if os.path.exists(folder_path):
                    results = predictor.predict_batch(folder_path)
                    if results:
                        print(f"\n✅ Successfully processed {len(results)} images!")
                    else:
                        print("❌ No images were processed successfully.")
                else:
                    print(f"❌ Folder not found: {folder_path}")
                    print("Please check the path and try again.")
            
            elif choice == '3':
                print("\n--- Prediction with Visualization ---")
                image_path = get_user_input("Enter path to image file: ")
                
                if not image_path:
                    print("No path provided.")
                    continue
                
                # Handle relative paths
                if not os.path.isabs(image_path) and not os.path.exists(image_path):
                    possible_paths = [
                        image_path,
                        os.path.join("data", image_path),
                        os.path.join("data", "val", "Moderate", image_path),
                        os.path.join("data", "val", "Mild", image_path),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            image_path = path
                            break
                
                if os.path.exists(image_path):
                    save_viz = get_user_input("Save visualization? (y/n): ").lower()
                    if save_viz.startswith('y'):
                        save_path = f"prediction_viz_{os.path.basename(image_path)}.png"
                        predictor.visualize_prediction(image_path, save_path)
                        print(f"✅ Visualization saved as: {save_path}")
                    else:
                        predictor.visualize_prediction(image_path)
                else:
                    print(f"❌ Image file not found: {image_path}")
                    print("Please check the path and try again.")
            
            elif choice == '4':
                print("\n👋 Thank you for using Diabetic Retinopathy Detection Tool!")
                print("Goodbye!")
                break
            
            else:
                print(f"❌ Invalid choice: '{choice}'")
                print("Please enter a number from 1 to 4.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()