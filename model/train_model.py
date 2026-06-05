# # import os
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# # from sklearn.metrics import classification_report, confusion_matrix
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from efficientnet_model import create_efficientnet_model, compile_model, unfreeze_model, CLASS_NAMES

# # # Set random seeds for reproducibility
# # np.random.seed(42)
# # tf.random.set_seed(42)

# # class DiabetinRetinopathyTrainer:
# #     """
# #     Trainer class for diabetic retinopathy detection model.
# #     """
    
# #     def __init__(self, data_dir, model_save_path="diabetic_retinopathy_model.h5"):
# #         """
# #         Initialize the trainer.
        
# #         Args:
# #             data_dir: Path to the data directory containing train and val folders
# #             model_save_path: Path to save the trained model
# #         """
# #         self.data_dir = data_dir
# #         self.model_save_path = model_save_path
# #         self.train_dir = os.path.join(data_dir, 'train')
# #         self.val_dir = os.path.join(data_dir, 'val')
# #         self.model = None
# #         self.history = None
        
# #         # Training parameters
# #         self.img_height = 224
# #         self.img_width = 224
# #         self.batch_size = 32
# #         self.num_classes = 5
        
# #     def create_data_generators(self):
# #         """
# #         Create data generators for training and validation.
        
# #         Returns:
# #             train_generator, validation_generator
# #         """
        
# #         # Data augmentation for training
# #         train_datagen = ImageDataGenerator(
# #             rescale=1./255,
# #             rotation_range=30,
# #             width_shift_range=0.2,
# #             height_shift_range=0.2,
# #             shear_range=0.2,
# #             zoom_range=0.2,
# #             horizontal_flip=True,
# #             vertical_flip=True,
# #             fill_mode='nearest',
# #             brightness_range=[0.8, 1.2],
# #             channel_shift_range=20.0
# #         )
        
# #         # Only rescaling for validation
# #         val_datagen = ImageDataGenerator(rescale=1./255)
        
# #         # Create generators
# #         train_generator = train_datagen.flow_from_directory(
# #             self.train_dir,
# #             target_size=(self.img_height, self.img_width),
# #             batch_size=self.batch_size,
# #             class_mode='categorical',
# #             shuffle=True,
# #             seed=42
# #         )
        
# #         validation_generator = val_datagen.flow_from_directory(
# #             self.val_dir,
# #             target_size=(self.img_height, self.img_width),
# #             batch_size=self.batch_size,
# #             class_mode='categorical',
# #             shuffle=False,
# #             seed=42
# #         )
        
# #         print(f"Training samples: {train_generator.samples}")
# #         print(f"Validation samples: {validation_generator.samples}")
# #         print(f"Class indices: {train_generator.class_indices}")
        
# #         return train_generator, validation_generator
    
# #     def create_callbacks(self):
# #         """
# #         Create training callbacks.
        
# #         Returns:
# #             List of callbacks
# #         """
        
# #         callbacks = [
# #             ModelCheckpoint(
# #                 self.model_save_path,
# #                 monitor='val_accuracy',
# #                 save_best_only=True,
# #                 save_weights_only=False,
# #                 mode='max',
# #                 verbose=1
# #             ),
# #             EarlyStopping(
# #                 monitor='val_loss',
# #                 patience=15,
# #                 restore_best_weights=True,
# #                 verbose=1
# #             ),
# #             ReduceLROnPlateau(
# #                 monitor='val_loss',
# #                 factor=0.2,
# #                 patience=5,
# #                 min_lr=1e-7,
# #                 verbose=1
# #             )
# #         ]
        
# #         return callbacks
    
# #     def train_model(self, initial_epochs=30, fine_tune_epochs=20):
# #         """
# #         Train the model with transfer learning and fine-tuning.
        
# #         Args:
# #             initial_epochs: Number of epochs for initial training
# #             fine_tune_epochs: Number of epochs for fine-tuning
        
# #         Returns:
# #             Training history
# #         """
        
# #         print("Creating data generators...")
# #         train_generator, validation_generator = self.create_data_generators()
        
# #         print("Creating model...")
# #         self.model = create_efficientnet_model(
# #             input_shape=(self.img_height, self.img_width, 3),
# #             num_classes=self.num_classes
# #         )
        
# #         self.model = compile_model(self.model, learning_rate=0.001)
        
# #         print("Model created successfully!")
# #         self.model.summary()
        
# #         callbacks = self.create_callbacks()
        
# #         # Phase 1: Train with frozen base model
# #         print("\n" + "="*50)
# #         print("PHASE 1: Training with frozen base model")
# #         print("="*50)
        
# #         history1 = self.model.fit(
# #             train_generator,
# #             steps_per_epoch=train_generator.samples // self.batch_size,
# #             epochs=initial_epochs,
# #             validation_data=validation_generator,
# #             validation_steps=validation_generator.samples // self.batch_size,
# #             callbacks=callbacks,
# #             verbose=1
# #         )
        
# #         # Phase 2: Fine-tuning
# #         print("\n" + "="*50)
# #         print("PHASE 2: Fine-tuning with unfrozen layers")
# #         print("="*50)
        
# #         # Unfreeze some layers for fine-tuning
# #         self.model = unfreeze_model(self.model, unfreeze_layers=50)
        
# #         # Recompile with lower learning rate
# #         self.model = compile_model(self.model, learning_rate=0.0001)
        
# #         # Update callbacks with new model path
# #         fine_tune_model_path = self.model_save_path.replace('.h5', '_fine_tuned.h5')
# #         callbacks[0] = ModelCheckpoint(
# #             fine_tune_model_path,
# #             monitor='val_accuracy',
# #             save_best_only=True,
# #             save_weights_only=False,
# #             mode='max',
# #             verbose=1
# #         )
        
# #         history2 = self.model.fit(
# #             train_generator,
# #             steps_per_epoch=train_generator.samples // self.batch_size,
# #             epochs=fine_tune_epochs,
# #             validation_data=validation_generator,
# #             validation_steps=validation_generator.samples // self.batch_size,
# #             callbacks=callbacks,
# #             verbose=1
# #         )
        
# #         # Combine histories
# #         self.history = self._combine_histories(history1, history2)
        
# #         # Save the final model
# #         self.model.save(fine_tune_model_path)
# #         self.model_save_path = fine_tune_model_path
        
# #         print(f"\nTraining completed! Model saved as {self.model_save_path}")
        
# #         return self.history
    
# #     def _combine_histories(self, hist1, hist2):
# #         """
# #         Combine two training histories.
        
# #         Args:
# #             hist1: First training history
# #             hist2: Second training history
        
# #         Returns:
# #             Combined history dictionary
# #         """
        
# #         combined_history = {}
        
# #         for key in hist1.history.keys():
# #             combined_history[key] = hist1.history[key] + hist2.history[key]
        
# #         return combined_history
    
# #     def plot_training_history(self, save_path=None):
# #         """
# #         Plot training history.
        
# #         Args:
# #             save_path: Path to save the plot
# #         """
        
# #         if self.history is None:
# #             print("No training history available. Train the model first.")
# #             return
        
# #         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
# #         # Plot accuracy
# #         axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy')
# #         axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy')
# #         axes[0, 0].set_title('Model Accuracy')
# #         axes[0, 0].set_xlabel('Epoch')
# #         axes[0, 0].set_ylabel('Accuracy')
# #         axes[0, 0].legend()
# #         axes[0, 0].grid(True)
        
# #         # Plot loss
# #         axes[0, 1].plot(self.history['loss'], label='Training Loss')
# #         axes[0, 1].plot(self.history['val_loss'], label='Validation Loss')
# #         axes[0, 1].set_title('Model Loss')
# #         axes[0, 1].set_xlabel('Epoch')
# #         axes[0, 1].set_ylabel('Loss')
# #         axes[0, 1].legend()
# #         axes[0, 1].grid(True)
        
# #         # Plot precision
# #         axes[1, 0].plot(self.history['precision'], label='Training Precision')
# #         axes[1, 0].plot(self.history['val_precision'], label='Validation Precision')
# #         axes[1, 0].set_title('Model Precision')
# #         axes[1, 0].set_xlabel('Epoch')
# #         axes[1, 0].set_ylabel('Precision')
# #         axes[1, 0].legend()
# #         axes[1, 0].grid(True)
        
# #         # Plot recall
# #         axes[1, 1].plot(self.history['recall'], label='Training Recall')
# #         axes[1, 1].plot(self.history['val_recall'], label='Validation Recall')
# #         axes[1, 1].set_title('Model Recall')
# #         axes[1, 1].set_xlabel('Epoch')
# #         axes[1, 1].set_ylabel('Recall')
# #         axes[1, 1].legend()
# #         axes[1, 1].grid(True)
        
# #         plt.tight_layout()
        
# #         if save_path:
# #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #             print(f"Training history plot saved as {save_path}")
        
# #         plt.show()
    
# #     def evaluate_model(self):
# #         """
# #         Evaluate the trained model on validation data.
        
# #         Returns:
# #             Evaluation results
# #         """
        
# #         if self.model is None:
# #             print("No model available. Train the model first.")
# #             return None
        
# #         print("Evaluating model...")
        
# #         # Create validation generator
# #         val_datagen = ImageDataGenerator(rescale=1./255)
# #         validation_generator = val_datagen.flow_from_directory(
# #             self.val_dir,
# #             target_size=(self.img_height, self.img_width),
# #             batch_size=self.batch_size,
# #             class_mode='categorical',
# #             shuffle=False
# #         )
        
# #         # Get predictions
# #         predictions = self.model.predict(validation_generator)
# #         y_pred = np.argmax(predictions, axis=1)
# #         y_true = validation_generator.classes
        
# #         # Classification report
# #         report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
# #         print("Classification Report:")
# #         print(report)
        
# #         # Confusion matrix
# #         cm = confusion_matrix(y_true, y_pred)
        
# #         # Plot confusion matrix
# #         plt.figure(figsize=(10, 8))
# #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #                     xticklabels=CLASS_NAMES,
# #                     yticklabels=CLASS_NAMES)
# #         plt.title('Confusion Matrix')
# #         plt.xlabel('Predicted Label')
# #         plt.ylabel('True Label')
# #         plt.tight_layout()
# #         plt.show()
        
# #         # Calculate accuracy
# #         accuracy = np.sum(y_pred == y_true) / len(y_true)
# #         print(f"\nValidation Accuracy: {accuracy:.4f}")
        
# #         return {
# #             'accuracy': accuracy,
# #             'classification_report': report,
# #             'confusion_matrix': cm,
# #             'predictions': predictions,
# #             'y_true': y_true,
# #             'y_pred': y_pred
# #         }

# # def main():
# #     """
# #     Main training function.
# #     """
    
# #     # Configuration
# #     DATA_DIR = "data"  # Update this path to your data directory
# #     MODEL_SAVE_PATH = "diabetic_retinopathy_model.h5"
    
# #     # Check if data directory exists
# #     if not os.path.exists(DATA_DIR):
# #         print(f"Data directory '{DATA_DIR}' not found!")
# #         print("Please make sure you have the following structure:")
# #         print("data/")
# #         print("├── train/")
# #         print("│   ├── 0_no_dr/")
# #         print("│   ├── 1_mild/")
# #         print("│   ├── 2_moderate/")
# #         print("│   ├── 3_severe/")
# #         print("│   └── 4_proliferative/")
# #         print("└── val/")
# #         print("    ├── 0_no_dr/")
# #         print("    ├── 1_mild/")
# #         print("    ├── 2_moderate/")
# #         print("    ├── 3_severe/")
# #         print("    └── 4_proliferative/")
# #         return
    
# #     # Initialize trainer
# #     trainer = DiabetinRetinopathyTrainer(DATA_DIR, MODEL_SAVE_PATH)
    
# #     # Train model
# #     print("Starting training...")
# #     history = trainer.train_model(initial_epochs=30, fine_tune_epochs=20)
    
# #     # Plot training history
# #     trainer.plot_training_history("training_history.png")
    
# #     # Evaluate model
# #     results = trainer.evaluate_model()
    
# #     print("Training and evaluation completed!")

# # if __name__ == "__main__":
# #     main()



# #yhaa h code

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from efficientnet_model import create_efficientnet_model, compile_model, unfreeze_model, CLASS_NAMES

# # Set random seeds for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

# class DiabetinRetinopathyTrainer:
#     """
#     Trainer class for diabetic retinopathy detection model.
#     """
    
#     def __init__(self, data_dir, model_save_path="diabetic_retinopathy_model.h5"):
#         """
#         Initialize the trainer.
        
#         Args:
#             data_dir: Path to the data directory containing train and val folders
#             model_save_path: Path to save the trained model
#         """
#         self.data_dir = data_dir
#         self.model_save_path = model_save_path
#         self.train_dir = os.path.join(data_dir, 'train')
#         self.val_dir = os.path.join(data_dir, 'val')
#         self.model = None
#         self.history = None
        
#         # Training parameters
#         self.img_height = 224
#         self.img_width = 224
#         self.batch_size = 32
#         self.num_classes = 5
        
#     def create_data_generators(self):
#         """
#         Create data generators for training and validation.
        
#         Returns:
#             train_generator, validation_generator
#         """
        
#         # Data augmentation for training
#         train_datagen = ImageDataGenerator(
#             rescale=1./255,
#             rotation_range=30,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True,
#             vertical_flip=True,
#             fill_mode='nearest',
#             brightness_range=[0.8, 1.2],
#             channel_shift_range=20.0
#         )
        
#         # Only rescaling for validation
#         val_datagen = ImageDataGenerator(rescale=1./255)
        
#         # Create generators
#         train_generator = train_datagen.flow_from_directory(
#             self.train_dir,
#             target_size=(self.img_height, self.img_width),
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             shuffle=True,
#             seed=42
#         )
        
#         validation_generator = val_datagen.flow_from_directory(
#             self.val_dir,
#             target_size=(self.img_height, self.img_width),
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             shuffle=False,
#             seed=42
#         )
        
#         print(f"Training samples: {train_generator.samples}")
#         print(f"Validation samples: {validation_generator.samples}")
#         print(f"Class indices: {train_generator.class_indices}")
        
#         return train_generator, validation_generator
    
#     def create_callbacks(self):
#         """
#         Create training callbacks.
        
#         Returns:
#             List of callbacks
#         """
        
#         callbacks = [
#             ModelCheckpoint(
#                 self.model_save_path,
#                 monitor='val_accuracy',
#                 save_best_only=True,
#                 save_weights_only=False,
#                 mode='max',
#                 verbose=1
#             ),
#             EarlyStopping(
#                 monitor='val_loss',
#                 patience=15,
#                 restore_best_weights=True,
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.2,
#                 patience=5,
#                 min_lr=1e-7,
#                 verbose=1
#             )
#         ]
        
#         return callbacks
    
#     def train_model(self, initial_epochs=30, fine_tune_epochs=20):
#         """
#         Train the model with transfer learning and fine-tuning.
        
#         Args:
#             initial_epochs: Number of epochs for initial training
#             fine_tune_epochs: Number of epochs for fine-tuning
        
#         Returns:
#             Training history
#         """
        
#         print("Creating data generators...")
#         train_generator, validation_generator = self.create_data_generators()
        
#         print("Creating model...")
#         self.model = create_efficientnet_model(
#             input_shape=(self.img_height, self.img_width, 3),
#             num_classes=self.num_classes
#         )
        
#         self.model = compile_model(self.model, learning_rate=0.001)
        
#         print("Model created successfully!")
#         self.model.summary()
        
#         callbacks = self.create_callbacks()
        
#         # Phase 1: Train with frozen base model
#         print("\n" + "="*50)
#         print("PHASE 1: Training with frozen base model")
#         print("="*50)
        
#         history1 = self.model.fit(
#             train_generator,
#             steps_per_epoch=train_generator.samples // self.batch_size,
#             epochs=initial_epochs,
#             validation_data=validation_generator,
#             validation_steps=validation_generator.samples // self.batch_size,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         # Phase 2: Fine-tuning
#         print("\n" + "="*50)
#         print("PHASE 2: Fine-tuning with unfrozen layers")
#         print("="*50)
        
#         # Unfreeze some layers for fine-tuning
#         self.model = unfreeze_model(self.model, unfreeze_layers=50)
        
#         # Recompile with lower learning rate
#         self.model = compile_model(self.model, learning_rate=0.0001)
        
#         # Update callbacks with new model path
#         fine_tune_model_path = self.model_save_path.replace('.h5', '_fine_tuned.h5')
#         callbacks[0] = ModelCheckpoint(
#             fine_tune_model_path,
#             monitor='val_accuracy',
#             save_best_only=True,
#             save_weights_only=False,
#             mode='max',
#             verbose=1
#         )
        
#         history2 = self.model.fit(
#             train_generator,
#             steps_per_epoch=train_generator.samples // self.batch_size,
#             epochs=fine_tune_epochs,
#             validation_data=validation_generator,
#             validation_steps=validation_generator.samples // self.batch_size,
#             callbacks=callbacks,
#             verbose=1
#         )
        
#         # Combine histories
#         self.history = self._combine_histories(history1, history2)
        
#         # Save the final model
#         self.model.save(fine_tune_model_path)
#         self.model_save_path = fine_tune_model_path
        
#         print(f"\nTraining completed! Model saved as {self.model_save_path}")
        
#         return self.history
    
#     def _combine_histories(self, hist1, hist2):
#         """
#         Combine two training histories.
        
#         Args:
#             hist1: First training history
#             hist2: Second training history
        
#         Returns:
#             Combined history dictionary
#         """
        
#         combined_history = {}
        
#         for key in hist1.history.keys():
#             combined_history[key] = hist1.history[key] + hist2.history[key]
        
#         return combined_history
    
#     def plot_training_history(self, save_path=None):
#         """
#         Plot training history.
        
#         Args:
#             save_path: Path to save the plot
#         """
        
#         if self.history is None:
#             print("No training history available. Train the model first.")
#             return
        
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
#         # Plot accuracy
#         axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy')
#         axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy')
#         axes[0, 0].set_title('Model Accuracy')
#         axes[0, 0].set_xlabel('Epoch')
#         axes[0, 0].set_ylabel('Accuracy')
#         axes[0, 0].legend()
#         axes[0, 0].grid(True)
        
#         # Plot loss
#         axes[0, 1].plot(self.history['loss'], label='Training Loss')
#         axes[0, 1].plot(self.history['val_loss'], label='Validation Loss')
#         axes[0, 1].set_title('Model Loss')
#         axes[0, 1].set_xlabel('Epoch')
#         axes[0, 1].set_ylabel('Loss')
#         axes[0, 1].legend()
#         axes[0, 1].grid(True)
        
#         # Plot precision
#         axes[1, 0].plot(self.history['precision'], label='Training Precision')
#         axes[1, 0].plot(self.history['val_precision'], label='Validation Precision')
#         axes[1, 0].set_title('Model Precision')
#         axes[1, 0].set_xlabel('Epoch')
#         axes[1, 0].set_ylabel('Precision')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True)
        
#         # Plot recall
#         axes[1, 1].plot(self.history['recall'], label='Training Recall')
#         axes[1, 1].plot(self.history['val_recall'], label='Validation Recall')
#         axes[1, 1].set_title('Model Recall')
#         axes[1, 1].set_xlabel('Epoch')
#         axes[1, 1].set_ylabel('Recall')
#         axes[1, 1].legend()
#         axes[1, 1].grid(True)
        
#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=300, bbox_inches='tight')
#             print(f"Training history plot saved as {save_path}")
        
#         plt.show()
    
#     def evaluate_model(self):
#         """
#         Evaluate the trained model on validation data.
        
#         Returns:
#             Evaluation results
#         """
        
#         if self.model is None:
#             print("No model available. Train the model first.")
#             return None
        
#         print("Evaluating model...")
        
#         # Create validation generator
#         val_datagen = ImageDataGenerator(rescale=1./255)
#         validation_generator = val_datagen.flow_from_directory(
#             self.val_dir,
#             target_size=(self.img_height, self.img_width),
#             batch_size=self.batch_size,
#             class_mode='categorical',
#             shuffle=False
#         )
        
#         # Get predictions
#         predictions = self.model.predict(validation_generator)
#         y_pred = np.argmax(predictions, axis=1)
#         y_true = validation_generator.classes
        
#         # Classification report
#         report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
#         print("Classification Report:")
#         print(report)
        
#         # Confusion matrix
#         cm = confusion_matrix(y_true, y_pred)
        
#         # Plot confusion matrix
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                     xticklabels=CLASS_NAMES,
#                     yticklabels=CLASS_NAMES)
#         plt.title('Confusion Matrix')
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.tight_layout()
#         plt.show()
        
#         # Calculate accuracy
#         accuracy = np.sum(y_pred == y_true) / len(y_true)
#         print(f"\nValidation Accuracy: {accuracy:.4f}")
        
#         return {
#             'accuracy': accuracy,
#             'classification_report': report,
#             'confusion_matrix': cm,
#             'predictions': predictions,
#             'y_true': y_true,
#             'y_pred': y_pred
#         }

# def main():
#     """
#     Main training function.
#     """
    
#     # Configuration
#     DATA_DIR = "data"  # Update this path to your data directory
#     MODEL_SAVE_PATH = "diabetic_retinopathy_model.h5"
    
#     # Check if data directory exists
#     if not os.path.exists(DATA_DIR):
#         print(f"Data directory '{DATA_DIR}' not found!")
#         print("Please make sure you have the following structure:")
#         print("data/")
#         print("├── train/")
#         print("│   ├── 0_no_dr/")
#         print("│   ├── 1_mild/")
#         print("│   ├── 2_moderate/")
#         print("│   ├── 3_severe/")
#         print("│   └── 4_proliferative/")
#         print("└── val/")
#         print("    ├── 0_no_dr/")
#         print("    ├── 1_mild/")
#         print("    ├── 2_moderate/")
#         print("    ├── 3_severe/")
#         print("    └── 4_proliferative/")
#         return
    
#     # Check if training folders have images
#     train_dir = os.path.join(DATA_DIR, 'train')
#     val_dir = os.path.join(DATA_DIR, 'val')
    
#     total_train_images = 0
#     total_val_images = 0
    
#     for class_folder in ['0_no_dr', '1_mild', '2_moderate', '3_severe', '4_proliferative']:
#         train_class_path = os.path.join(train_dir, class_folder)
#         val_class_path = os.path.join(val_dir, class_folder)
        
#         if os.path.exists(train_class_path):
#             train_count = len([f for f in os.listdir(train_class_path) 
#                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
#             total_train_images += train_count
#             print(f"Train {class_folder}: {train_count} images")
        
#         if os.path.exists(val_class_path):
#             val_count = len([f for f in os.listdir(val_class_path) 
#                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
#             total_val_images += val_count
#             print(f"Val {class_folder}: {val_count} images")
    
#     print(f"\nTotal training images: {total_train_images}")
#     print(f"Total validation images: {total_val_images}")
    
#     if total_train_images == 0:
#         print("No training images found! Please add images to the training folders.")
#         return
    
#     # Initialize trainer
#     trainer = DiabetinRetinopathyTrainer(DATA_DIR, MODEL_SAVE_PATH)
    
#     # Ask user which model to use
#     print("\nChoose model type:")
#     print("1. EfficientNet (recommended, but may have compatibility issues)")
#     print("2. Simple CNN (more compatible, faster training)")
    
#     try:
#         choice = input("Enter your choice (1 or 2): ").strip()
#         use_simple_cnn = (choice == '2')
#     except:
#         print("Using Simple CNN as default...")
#         use_simple_cnn = True
    
#     # Determine training epochs based on dataset size
#     if total_train_images < 1000:
#         initial_epochs = 20
#         fine_tune_epochs = 10
#     else:
#         initial_epochs = 30
#         fine_tune_epochs = 20
    
#     print(f"\nTraining configuration:")
#     print(f"Model: {'Simple CNN' if use_simple_cnn else 'EfficientNet'}")
#     print(f"Initial epochs: {initial_epochs}")
#     print(f"Fine-tune epochs: {fine_tune_epochs}")
#     print(f"Batch size: {trainer.batch_size}")
    
#     # Train model
#     print("\nStarting training...")
#     try:
#         history = trainer.train_model(
#             initial_epochs=initial_epochs, 
#             fine_tune_epochs=fine_tune_epochs,
#             use_simple_cnn=use_simple_cnn
#         )
        
#         # Plot training history
#         trainer.plot_training_history("training_history.png")
        
#         # Evaluate model
#         print("\nEvaluating model...")
#         results = trainer.evaluate_model()
        
#         print("Training and evaluation completed!")
        
#     except Exception as e:
#         print(f"Training failed with error: {e}")
#         print("Please check your dataset and try again.")
#         print("If the error persists, try using the Simple CNN model.")

# if __name__ == "__main__":
#     main()

# # import os
# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# # from sklearn.metrics import classification_report, confusion_matrix
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from efficientnet_model import create_efficientnet_model, compile_model, unfreeze_model, CLASS_NAMES

# # # Set random seeds for reproducibility
# # np.random.seed(42)
# # tf.random.set_seed(42)

# # class DiabetinRetinopathyTrainer:
# #     """
# #     Trainer class for diabetic retinopathy detection model.
# #     """
    
# #     def __init__(self, data_dir, model_save_path="diabetic_retinopathy_model.h5"):
# #         """
# #         Initialize the trainer.
        
# #         Args:
# #             data_dir: Path to the data directory containing train and val folders
# #             model_save_path: Path to save the trained model
# #         """
# #         self.data_dir = data_dir
# #         self.model_save_path = model_save_path
# #         self.train_dir = os.path.join(data_dir, 'train')
# #         self.val_dir = os.path.join(data_dir, 'val')
# #         self.model = None
# #         self.history = None
        
# #         # Training parameters
# #         self.img_height = 224
# #         self.img_width = 224
# #         self.batch_size = 32
# #         self.num_classes = 5
        
# #     def create_data_generators(self):
# #         """
# #         Create data generators for training and validation.
        
# #         Returns:
# #             train_generator, validation_generator
# #         """
        
# #         # Data augmentation for training
# #         train_datagen = ImageDataGenerator(
# #             rescale=1./255,
# #             rotation_range=30,
# #             width_shift_range=0.2,
# #             height_shift_range=0.2,
# #             shear_range=0.2,
# #             zoom_range=0.2,
# #             horizontal_flip=True,
# #             vertical_flip=True,
# #             fill_mode='nearest',
# #             brightness_range=[0.8, 1.2],
# #             channel_shift_range=20.0
# #         )
        
# #         # Only rescaling for validation
# #         val_datagen = ImageDataGenerator(rescale=1./255)
        
# #         # Create generators
# #         train_generator = train_datagen.flow_from_directory(
# #             self.train_dir,
# #             target_size=(self.img_height, self.img_width),
# #             batch_size=self.batch_size,
# #             class_mode='categorical',
# #             shuffle=True,
# #             seed=42
# #         )
        
# #         validation_generator = val_datagen.flow_from_directory(
# #             self.val_dir,
# #             target_size=(self.img_height, self.img_width),
# #             batch_size=self.batch_size,
# #             class_mode='categorical',
# #             shuffle=False,
# #             seed=42
# #         )
        
# #         print(f"Training samples: {train_generator.samples}")
# #         print(f"Validation samples: {validation_generator.samples}")
# #         print(f"Class indices: {train_generator.class_indices}")
        
# #         return train_generator, validation_generator
    
# #     def create_callbacks(self):
# #         """
# #         Create training callbacks.
        
# #         Returns:
# #             List of callbacks
# #         """
        
# #         callbacks = [
# #             ModelCheckpoint(
# #                 self.model_save_path,
# #                 monitor='val_accuracy',
# #                 save_best_only=True,
# #                 save_weights_only=False,
# #                 mode='max',
# #                 verbose=1
# #             ),
# #             EarlyStopping(
# #                 monitor='val_loss',
# #                 patience=15,
# #                 restore_best_weights=True,
# #                 verbose=1
# #             ),
# #             ReduceLROnPlateau(
# #                 monitor='val_loss',
# #                 factor=0.2,
# #                 patience=5,
# #                 min_lr=1e-7,
# #                 verbose=1
# #             )
# #         ]
        
# #         return callbacks
    
# #     def train_model(self, initial_epochs=30, fine_tune_epochs=20):
# #         """
# #         Train the model with transfer learning and fine-tuning.
        
# #         Args:
# #             initial_epochs: Number of epochs for initial training
# #             fine_tune_epochs: Number of epochs for fine-tuning
        
# #         Returns:
# #             Training history
# #         """
        
# #         print("Creating data generators...")
# #         train_generator, validation_generator = self.create_data_generators()
        
# #         print("Creating model...")
# #         self.model = create_efficientnet_model(
# #             input_shape=(self.img_height, self.img_width, 3),
# #             num_classes=self.num_classes
# #         )
        
# #         self.model = compile_model(self.model, learning_rate=0.001)
        
# #         print("Model created successfully!")
# #         self.model.summary()
        
# #         callbacks = self.create_callbacks()
        
# #         # Phase 1: Train with frozen base model
# #         print("\n" + "="*50)
# #         print("PHASE 1: Training with frozen base model")
# #         print("="*50)
        
# #         history1 = self.model.fit(
# #             train_generator,
# #             steps_per_epoch=train_generator.samples // self.batch_size,
# #             epochs=initial_epochs,
# #             validation_data=validation_generator,
# #             validation_steps=validation_generator.samples // self.batch_size,
# #             callbacks=callbacks,
# #             verbose=1
# #         )
        
# #         # Phase 2: Fine-tuning
# #         print("\n" + "="*50)
# #         print("PHASE 2: Fine-tuning with unfrozen layers")
# #         print("="*50)
        
# #         # Unfreeze some layers for fine-tuning
# #         self.model = unfreeze_model(self.model, unfreeze_layers=50)
        
# #         # Recompile with lower learning rate
# #         self.model = compile_model(self.model, learning_rate=0.0001)
        
# #         # Update callbacks with new model path
# #         fine_tune_model_path = self.model_save_path.replace('.h5', '_fine_tuned.h5')
# #         callbacks[0] = ModelCheckpoint(
# #             fine_tune_model_path,
# #             monitor='val_accuracy',
# #             save_best_only=True,
# #             save_weights_only=False,
# #             mode='max',
# #             verbose=1
# #         )
        
# #         history2 = self.model.fit(
# #             train_generator,
# #             steps_per_epoch=train_generator.samples // self.batch_size,
# #             epochs=fine_tune_epochs,
# #             validation_data=validation_generator,
# #             validation_steps=validation_generator.samples // self.batch_size,
# #             callbacks=callbacks,
# #             verbose=1
# #         )
        
# #         # Combine histories
# #         self.history = self._combine_histories(history1, history2)
        
# #         # Save the final model
# #         self.model.save(fine_tune_model_path)
# #         self.model_save_path = fine_tune_model_path
        
# #         print(f"\nTraining completed! Model saved as {self.model_save_path}")
        
# #         return self.history
    
# #     def _combine_histories(self, hist1, hist2):
# #         """
# #         Combine two training histories.
        
# #         Args:
# #             hist1: First training history
# #             hist2: Second training history
        
# #         Returns:
# #             Combined history dictionary
# #         """
        
# #         combined_history = {}
        
# #         for key in hist1.history.keys():
# #             combined_history[key] = hist1.history[key] + hist2.history[key]
        
# #         return combined_history
    
# #     def plot_training_history(self, save_path=None):
# #         """
# #         Plot training history.
        
# #         Args:
# #             save_path: Path to save the plot
# #         """
        
# #         if self.history is None:
# #             print("No training history available. Train the model first.")
# #             return
        
# #         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
# #         # Plot accuracy
# #         axes[0, 0].plot(self.history['accuracy'], label='Training Accuracy')
# #         axes[0, 0].plot(self.history['val_accuracy'], label='Validation Accuracy')
# #         axes[0, 0].set_title('Model Accuracy')
# #         axes[0, 0].set_xlabel('Epoch')
# #         axes[0, 0].set_ylabel('Accuracy')
# #         axes[0, 0].legend()
# #         axes[0, 0].grid(True)
        
# #         # Plot loss
# #         axes[0, 1].plot(self.history['loss'], label='Training Loss')
# #         axes[0, 1].plot(self.history['val_loss'], label='Validation Loss')
# #         axes[0, 1].set_title('Model Loss')
# #         axes[0, 1].set_xlabel('Epoch')
# #         axes[0, 1].set_ylabel('Loss')
# #         axes[0, 1].legend()
# #         axes[0, 1].grid(True)
        
# #         # Plot precision
# #         axes[1, 0].plot(self.history['precision'], label='Training Precision')
# #         axes[1, 0].plot(self.history['val_precision'], label='Validation Precision')
# #         axes[1, 0].set_title('Model Precision')
# #         axes[1, 0].set_xlabel('Epoch')
# #         axes[1, 0].set_ylabel('Precision')
# #         axes[1, 0].legend()
# #         axes[1, 0].grid(True)
        
# #         # Plot recall
# #         axes[1, 1].plot(self.history['recall'], label='Training Recall')
# #         axes[1, 1].plot(self.history['val_recall'], label='Validation Recall')
# #         axes[1, 1].set_title('Model Recall')
# #         axes[1, 1].set_xlabel('Epoch')
# #         axes[1, 1].set_ylabel('Recall')
# #         axes[1, 1].legend()
# #         axes[1, 1].grid(True)
        
# #         plt.tight_layout()
        
# #         if save_path:
# #             plt.savefig(save_path, dpi=300, bbox_inches='tight')
# #             print(f"Training history plot saved as {save_path}")
        
# #         plt.show()
    
# #     def evaluate_model(self):
# #         """
# #         Evaluate the trained model on validation data.
        
# #         Returns:
# #             Evaluation results
# #         """
        
# #         if self.model is None:
# #             print("No model available. Train the model first.")
# #             return None
        
# #         print("Evaluating model...")
        
# #         # Create validation generator
# #         val_datagen = ImageDataGenerator(rescale=1./255)
# #         validation_generator = val_datagen.flow_from_directory(
# #             self.val_dir,
# #             target_size=(self.img_height, self.img_width),
# #             batch_size=self.batch_size,
# #             class_mode='categorical',
# #             shuffle=False
# #         )
        
# #         # Get predictions
# #         predictions = self.model.predict(validation_generator)
# #         y_pred = np.argmax(predictions, axis=1)
# #         y_true = validation_generator.classes
        
# #         # Classification report
# #         report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
# #         print("Classification Report:")
# #         print(report)
        
# #         # Confusion matrix
# #         cm = confusion_matrix(y_true, y_pred)
        
# #         # Plot confusion matrix
# #         plt.figure(figsize=(10, 8))
# #         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
# #                     xticklabels=CLASS_NAMES,
# #                     yticklabels=CLASS_NAMES)
# #         plt.title('Confusion Matrix')
# #         plt.xlabel('Predicted Label')
# #         plt.ylabel('True Label')
# #         plt.tight_layout()
# #         plt.show()
        
# #         # Calculate accuracy
# #         accuracy = np.sum(y_pred == y_true) / len(y_true)
# #         print(f"\nValidation Accuracy: {accuracy:.4f}")
        
# #         return {
# #             'accuracy': accuracy,
# #             'classification_report': report,
# #             'confusion_matrix': cm,
# #             'predictions': predictions,
# #             'y_true': y_true,
# #             'y_pred': y_pred
# #         }

# # def main():
# #     """
# #     Main training function.
# #     """
    
# #     # Configuration
# #     DATA_DIR = "data"  # Update this path to your data directory
# #     MODEL_SAVE_PATH = "diabetic_retinopathy_model.h5"
    
# #     # Check if data directory exists
# #     if not os.path.exists(DATA_DIR):
# #         print(f"Data directory '{DATA_DIR}' not found!")
# #         print("Please make sure you have the following structure:")
# #         print("data/")
# #         print("├── train/")
# #         print("│   ├── DR/")
# #         print("│   └── No_DR/")
# #         print("└── val/")
# #         print("    ├── DR/")
# #         print("    └── No_DR/")
# #         return
    
# #     # Check if training folders have images
# #     train_dir = os.path.join(DATA_DIR, 'train')
# #     val_dir = os.path.join(DATA_DIR, 'val')
    
# #     total_train_images = 0
# #     total_val_images = 0
    
# #     for class_folder in ['0_no_dr', '1_mild', '2_moderate', '3_severe', '4_proliferative']:
# #         train_class_path = os.path.join(train_dir, class_folder)
# #         val_class_path = os.path.join(val_dir, class_folder)
        
# #         if os.path.exists(train_class_path):
# #             train_count = len([f for f in os.listdir(train_class_path) 
# #                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
# #             total_train_images += train_count
# #             print(f"Train {class_folder}: {train_count} images")
        
# #         if os.path.exists(val_class_path):
# #             val_count = len([f for f in os.listdir(val_class_path) 
# #                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
# #             total_val_images += val_count
# #             print(f"Val {class_folder}: {val_count} images")
    
# #     print(f"\nTotal training images: {total_train_images}")
# #     print(f"Total validation images: {total_val_images}")
    
# #     if total_train_images == 0:
# #         print("No training images found! Please add images to the training folders.")
# #         return
    
# #     # Initialize trainer
# #     trainer = DiabetinRetinopathyTrainer(DATA_DIR, MODEL_SAVE_PATH)
    
# #     # Ask user which model to use
# #     print("\nChoose model type:")
# #     print("1. EfficientNet (recommended, but may have compatibility issues)")
# #     print("2. Simple CNN (more compatible, faster training)")
    
# #     try:
# #         choice = input("Enter your choice (1 or 2): ").strip()
# #         use_simple_cnn = (choice == '2')
# #     except:
# #         print("Using Simple CNN as default...")
# #         use_simple_cnn = True
    
# #     # Determine training epochs based on dataset size
# #     if total_train_images < 1000:
# #         initial_epochs = 20
# #         fine_tune_epochs = 10
# #     else:
# #         initial_epochs = 30
# #         fine_tune_epochs = 20
    
# #     print(f"\nTraining configuration:")
# #     print(f"Model: {'Simple CNN' if use_simple_cnn else 'EfficientNet'}")
# #     print(f"Initial epochs: {initial_epochs}")
# #     print(f"Fine-tune epochs: {fine_tune_epochs}")
# #     print(f"Batch size: {trainer.batch_size}")
    
# #     # Train model
# #     print("\nStarting training...")
# #     try:
# #         history = trainer.train_model(
# #             initial_epochs=initial_epochs, 
# #             fine_tune_epochs=fine_tune_epochs,
# #             use_simple_cnn=use_simple_cnn
# #         )
        
# #         # Plot training history
# #         trainer.plot_training_history("training_history.png")
        
# #         # Evaluate model
# #         print("\nEvaluating model...")
# #         results = trainer.evaluate_model()
        
# #         print("Training and evaluation completed!")
        
# #     except Exception as e:
# #         print(f"Training failed with error: {e}")
# #         print("Please check your dataset and try again.")
# #         print("If the error persists, try using the Simple CNN model.")

# # if __name__ == "__main__":
# #     main()







# #!/usr/bin/env python3
# """
# train_model.py - Diabetic Retinopathy Detection Training Script

# This file contains the main training logic for the diabetic retinopathy detection model.
# Fixed the parameter error and improved model handling.
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# class DiabetinRetinopathyTrainer:
#     def __init__(self, data_path, img_size=(224, 224), num_classes=5):
#         self.data_path = data_path
#         self.img_size = img_size
#         self.num_classes = num_classes
#         self.model = None
        
#     def create_efficientnet_model(self):
#         """Create EfficientNet-based model"""
#         try:
#             # Use EfficientNetB0 as base model
#             base_model = tf.keras.applications.EfficientNetB0(
#                 input_shape=(*self.img_size, 3),
#                 include_top=False,
#                 weights='imagenet'
#             )
            
#             # Freeze base model initially
#             base_model.trainable = False
            
#             # Add custom classification head
#             model = keras.Sequential([
#                 base_model,
#                 layers.GlobalAveragePooling2D(),
#                 layers.Dropout(0.3),
#                 layers.Dense(128, activation='relu'),
#                 layers.Dropout(0.2),
#                 layers.Dense(self.num_classes, activation='softmax')
#             ])
            
#             return model, True  # Success flag
            
#         except Exception as e:
#             print(f"EfficientNet creation failed: {e}")
#             return None, False
    
#     def create_simple_cnn_model(self):
#         """Create simple CNN model as fallback"""
#         model = keras.Sequential([
#             layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Flatten(),
#             layers.Dropout(0.5),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.3),
#             layers.Dense(self.num_classes, activation='softmax')
#         ])
        
#         return model, True
    
#     def prepare_data(self, batch_size=32):
#         """Prepare training data"""
#         # Data augmentation for training
#         train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             horizontal_flip=True,
#             zoom_range=0.2,
#             validation_split=0.2
#         )
        
#         # Only rescaling for validation
#         val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             validation_split=0.2
#         )
        
#         # Create training generator
#         train_generator = train_datagen.flow_from_directory(
#             self.data_path,
#             target_size=self.img_size,
#             batch_size=batch_size,
#             class_mode='categorical',
#             subset='training'
#         )
        
#         # Create validation generator
#         val_generator = val_datagen.flow_from_directory(
#             self.data_path,
#             target_size=self.img_size,
#             batch_size=batch_size,
#             class_mode='categorical',
#             subset='validation'
#         )
        
#         return train_generator, val_generator
    
#     def train_model(self, model_type='efficientnet', initial_epochs=30, 
#                    fine_tune_epochs=20, batch_size=32):
#         """
#         Train the model with specified parameters
        
#         Args:
#             model_type (str): 'efficientnet' or 'simple_cnn'
#             initial_epochs (int): Number of initial training epochs
#             fine_tune_epochs (int): Number of fine-tuning epochs (EfficientNet only)
#             batch_size (int): Batch size for training
            
#         Returns:
#             bool: True if training successful, False otherwise
#         """
#         print(f"Training configuration:")
#         print(f"Model: {model_type}")
#         print(f"Initial epochs: {initial_epochs}")
#         print(f"Fine-tune epochs: {fine_tune_epochs}")
#         print(f"Batch size: {batch_size}")
#         print("Starting training...")
        
#         try:
#             # Create model based on type
#             if model_type.lower() == 'efficientnet':
#                 self.model, success = self.create_efficientnet_model()
#                 if not success:
#                     print("EfficientNet failed, falling back to Simple CNN")
#                     self.model, success = self.create_simple_cnn_model()
#             else:
#                 self.model, success = self.create_simple_cnn_model()
            
#             if not success:
#                 raise Exception("Failed to create model")
            
#             # Prepare data
#             train_gen, val_gen = self.prepare_data(batch_size)
            
#             # Compile model
#             self.model.compile(
#                 optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy']
#             )
            
#             # Print model summary
#             print("\nModel Summary:")
#             self.model.summary()
            
#             # Callbacks
#             callbacks = [
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_accuracy',
#                     patience=5,
#                     restore_best_weights=True
#                 ),
#                 keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.2,
#                     patience=3,
#                     min_lr=1e-7
#                 ),
#                 keras.callbacks.ModelCheckpoint(
#                     'best_model.h5',
#                     monitor='val_accuracy',
#                     save_best_only=True,
#                     verbose=1
#                 )
#             ]
            
#             # Initial training
#             print("\nPhase 1: Initial training...")
#             history1 = self.model.fit(
#                 train_gen,
#                 epochs=initial_epochs,
#                 validation_data=val_gen,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             # Fine-tuning (only for EfficientNet)
#             if model_type.lower() == 'efficientnet' and fine_tune_epochs > 0:
#                 print("\nPhase 2: Fine-tuning...")
#                 # Unfreeze some layers for fine-tuning
#                 base_model = self.model.layers[0]
#                 base_model.trainable = True
                
#                 # Fine-tune from this layer onwards
#                 fine_tune_at = 100
#                 for layer in base_model.layers[:fine_tune_at]:
#                     layer.trainable = False
                
#                 # Use lower learning rate for fine-tuning
#                 self.model.compile(
#                     optimizer=keras.optimizers.Adam(1e-5/10),
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy']
#                 )
                
#                 history2 = self.model.fit(
#                     train_gen,
#                     epochs=initial_epochs + fine_tune_epochs,
#                     initial_epoch=len(history1.history['loss']),
#                     validation_data=val_gen,
#                     callbacks=callbacks,
#                     verbose=1
#                 )
            
#             print("\n" + "="*50)
#             print("Training completed successfully!")
#             print("="*50)
#             return True
            
#         except Exception as e:
#             print(f"\nTraining failed with error: {e}")
#             print("Please check your dataset and try again.")
#             if "efficientnet" in model_type.lower():
#                 print("If the error persists, try using the Simple CNN model.")
#             return False
    
#     def save_model(self, filepath='diabetic_retinopathy_model.h5'):
#         """Save the trained model"""
#         if self.model is not None:
#             self.model.save(filepath)
#             print(f"Model saved to {filepath}")
#         else:
#             print("No model to save!")
    
#     def evaluate_model(self, test_data_path=None):
#         """Evaluate the model"""
#         if self.model is None:
#             print("No model to evaluate!")
#             return
        
#         # Use validation data if no test data provided
#         if test_data_path is None:
#             test_data_path = self.data_path
        
#         test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#         test_generator = test_datagen.flow_from_directory(
#             test_data_path,
#             target_size=self.img_size,
#             batch_size=32,
#             class_mode='categorical',
#             shuffle=False
#         )
        
#         # Evaluate
#         print("\nEvaluating model...")
#         loss, accuracy = self.model.evaluate(test_generator, verbose=1)
#         print(f"\nTest Results:")
#         print(f"Test Accuracy: {accuracy:.4f}")
#         print(f"Test Loss: {loss:.4f}")


# def main():
#     """Main training function"""
#     print("Diabetic Retinopathy Detection Training")
#     print("=" * 40)
    
#     # Configuration
#     DATA_PATH = "data"  # Update this to your dataset path
    
#     # Check if dataset exists
#     if not os.path.exists(DATA_PATH):
#         print(f"Error: Dataset path '{DATA_PATH}' does not exist!")
#         print("Please update the DATA_PATH variable with your correct dataset path.")
#         return
    
#     # Get user choice
#     print("Choose model type:")
#     print("1. EfficientNet (recommended, but may have compatibility issues)")
#     print("2. Simple CNN (more compatible, faster training)")
    
#     try:
#         choice = input("Enter your choice (1 or 2): ").strip()
        
#         # Set model type based on choice
#         if choice == '1':
#             model_type = 'efficientnet'
#         elif choice == '2':
#             model_type = 'simple_cnn'
#         else:
#             print("Invalid choice, using EfficientNet as default")
#             model_type = 'efficientnet'
        
#         # Initialize trainer
#         trainer = DiabetinRetinopathyTrainer(DATA_PATH)
        
#         # Train model
#         success = trainer.train_model(
#             model_type=model_type,
#             initial_epochs=30,
#             fine_tune_epochs=20,
#             batch_size=32
#         )
        
#         if success:
#             # Save the model
#             trainer.save_model('diabetic_retinopathy_model.h5')
            
#             # Evaluate the model
#             trainer.evaluate_model()
#         else:
#             print("\nTraining failed. Please check your dataset and try again.")
#             print("If the error persists, try using the Simple CNN model.")
    
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user.")
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         print("Please check your dataset path and try again.")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# train_model.py - Diabetic Retinopathy Detection Training Script

# This file contains the main training logic for the diabetic retinopathy detection model.
# Fixed the parameter error and improved model handling.
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# class DiabetinRetinopathyTrainer:
#     def __init__(self, data_path, img_size=(224, 224), num_classes=5):
#         self.data_path = data_path
#         self.img_size = img_size
#         self.num_classes = num_classes
#         self.model = None
        
#     def create_efficientnet_model(self):
#         """Create EfficientNet-based model"""
#         try:
#             # Use EfficientNetB0 as base model
#             base_model = tf.keras.applications.EfficientNetB0(
#                 input_shape=(*self.img_size, 3),
#                 include_top=False,
#                 weights='imagenet'
#             )
            
#             # Freeze base model initially
#             base_model.trainable = False
            
#             # Add custom classification head
#             model = keras.Sequential([
#                 base_model,
#                 layers.GlobalAveragePooling2D(),
#                 layers.Dropout(0.3),
#                 layers.Dense(128, activation='relu'),
#                 layers.Dropout(0.2),
#                 layers.Dense(self.num_classes, activation='softmax')
#             ])
            
#             return model, True  # Success flag
            
#         except Exception as e:
#             print(f"EfficientNet creation failed: {e}")
#             return None, False
    
#     def create_simple_cnn_model(self):
#         """Create simple CNN model as fallback"""
#         model = keras.Sequential([
#             layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Flatten(),
#             layers.Dropout(0.5),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.3),
#             layers.Dense(self.num_classes, activation='softmax')
#         ])
        
#         return model, True
    
#     def prepare_data(self, batch_size=32):
#         """Prepare training data"""
#         # Data augmentation for training
#         train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             rotation_range=20,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             horizontal_flip=True,
#             zoom_range=0.2,
#             validation_split=0.2
#         )
        
#         # Only rescaling for validation
#         val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             validation_split=0.2
#         )
        
#         # Create training generator
#         train_generator = train_datagen.flow_from_directory(
#             self.data_path,
#             target_size=self.img_size,
#             batch_size=batch_size,
#             class_mode='categorical',
#             subset='training'
#         )
        
#         # Create validation generator
#         val_generator = val_datagen.flow_from_directory(
#             self.data_path,
#             target_size=self.img_size,
#             batch_size=batch_size,
#             class_mode='categorical',
#             subset='validation'
#         )
        
#         return train_generator, val_generator
    
#     def train_model(self, model_type='efficientnet', initial_epochs=30, 
#                    fine_tune_epochs=20, batch_size=32):
#         """
#         Train the model with specified parameters
        
#         Args:
#             model_type (str): 'efficientnet' or 'simple_cnn'
#             initial_epochs (int): Number of initial training epochs
#             fine_tune_epochs (int): Number of fine-tuning epochs (EfficientNet only)
#             batch_size (int): Batch size for training
            
#         Returns:
#             bool: True if training successful, False otherwise
#         """
#         print(f"Training configuration:")
#         print(f"Model: {model_type}")
#         print(f"Initial epochs: {initial_epochs}")
#         print(f"Fine-tune epochs: {fine_tune_epochs}")
#         print(f"Batch size: {batch_size}")
#         print("Starting training...")
        
#         try:
#             # Create model based on type
#             if model_type.lower() == 'efficientnet':
#                 self.model, success = self.create_efficientnet_model()
#                 if not success:
#                     print("EfficientNet failed, falling back to Simple CNN")
#                     self.model, success = self.create_simple_cnn_model()
#             else:
#                 self.model, success = self.create_simple_cnn_model()
            
#             if not success:
#                 raise Exception("Failed to create model")
            
#             # Prepare data
#             train_gen, val_gen = self.prepare_data(batch_size)
            
#             # Compile model
#             self.model.compile(
#                 optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy']
#             )
            
#             # Print model summary
#             print("\nModel Summary:")
#             self.model.summary()
            
#             # Callbacks
#             callbacks = [
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_accuracy',
#                     patience=5,
#                     restore_best_weights=True
#                 ),
#                 keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.2,
#                     patience=3,
#                     min_lr=1e-7
#                 ),
#                 keras.callbacks.ModelCheckpoint(
#                     'best_model.h5',
#                     monitor='val_accuracy',
#                     save_best_only=True,
#                     verbose=1
#                 )
#             ]
            
#             # Initial training
#             print("\nPhase 1: Initial training...")
#             history1 = self.model.fit(
#                 train_gen,
#                 epochs=initial_epochs,
#                 validation_data=val_gen,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             # Fine-tuning (only for EfficientNet)
#             if model_type.lower() == 'efficientnet' and fine_tune_epochs > 0:
#                 print("\nPhase 2: Fine-tuning...")
#                 # Unfreeze some layers for fine-tuning
#                 base_model = self.model.layers[0]
#                 base_model.trainable = True
                
#                 # Fine-tune from this layer onwards
#                 fine_tune_at = 100
#                 for layer in base_model.layers[:fine_tune_at]:
#                     layer.trainable = False
                
#                 # Use lower learning rate for fine-tuning
#                 self.model.compile(
#                     optimizer=keras.optimizers.Adam(1e-5/10),
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy']
#                 )
                
#                 history2 = self.model.fit(
#                     train_gen,
#                     epochs=initial_epochs + fine_tune_epochs,
#                     initial_epoch=len(history1.history['loss']),
#                     validation_data=val_gen,
#                     callbacks=callbacks,
#                     verbose=1
#                 )
            
#             print("\n" + "="*50)
#             print("Training completed successfully!")
#             print("="*50)
#             return True
            
#         except Exception as e:
#             print(f"\nTraining failed with error: {e}")
#             print("Please check your dataset and try again.")
#             if "efficientnet" in model_type.lower():
#                 print("If the error persists, try using the Simple CNN model.")
#             return False
    
#     def save_model(self, filepath='diabetic_retinopathy_model.h5'):
#         """Save the trained model"""
#         if self.model is not None:
#             self.model.save(filepath)
#             print(f"Model saved to {filepath}")
#         else:
#             print("No model to save!")
    
#     def evaluate_model(self, test_data_path=None):
#         """Evaluate the model"""
#         if self.model is None:
#             print("No model to evaluate!")
#             return
        
#         # Use validation data if no test data provided
#         if test_data_path is None:
#             test_data_path = self.data_path
        
#         test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#         test_generator = test_datagen.flow_from_directory(
#             test_data_path,
#             target_size=self.img_size,
#             batch_size=32,
#             class_mode='categorical',
#             shuffle=False
#         )
        
#         # Evaluate
#         print("\nEvaluating model...")
#         loss, accuracy = self.model.evaluate(test_generator, verbose=1)
#         print(f"\nTest Results:")
#         print(f"Test Accuracy: {accuracy:.4f}")
#         print(f"Test Loss: {loss:.4f}")


# def main():
#     """Main training function"""
#     print("Diabetic Retinopathy Detection Training")
#     print("=" * 40)
    
#     # Configuration
#     DATA_PATH = "data"  # Update this to match your folder structure
    
#     # Check dataset structure
#     print("Checking dataset structure...")
#     if not check_dataset_structure(DATA_PATH):
#         print("Please fix your dataset structure and try again.")
#         return
    
#     # Get user choice
#     print("Choose model type:")
#     print("1. EfficientNet (recommended, but may have compatibility issues)")
#     print("2. Simple CNN (more compatible, faster training)")
    
#     try:
#         choice = input("Enter your choice (1 or 2): ").strip()
        
#         # Set model type based on choice
#         if choice == '1':
#             model_type = 'efficientnet'
#         elif choice == '2':
#             model_type = 'simple_cnn'
#         else:
#             print("Invalid choice, using EfficientNet as default")
#             model_type = 'efficientnet'
        
#         # Initialize trainer
#         trainer = DiabetinRetinopathyTrainer(DATA_PATH)
        
#         # Train model
#         success = trainer.train_model(
#             model_type=model_type,
#             initial_epochs=30,
#             fine_tune_epochs=20,
#             batch_size=32
#         )
        
#         if success:
#             # Save the model
#             trainer.save_model('diabetic_retinopathy_model.h5')
            
#             # Evaluate the model
#             trainer.evaluate_model()
#         else:
#             print("\nTraining failed. Please check your dataset and try again.")
#             print("If the error persists, try using the Simple CNN model.")
    
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user.")
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         print("Please check your dataset path and try again.")


# if __name__ == "__main__":
#     main()





# #!/usr/bin/env python3
# """
# train_model.py - Diabetic Retinopathy Detection Training Script
# Fixed version with proper function definitions and error handling
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# import os
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# class DiabetinRetinopathyTrainer:
#     def __init__(self, data_path, img_size=(224, 224), num_classes=None):
#         self.data_path = data_path
#         self.img_size = img_size
#         self.num_classes = num_classes
#         self.model = None
#         self.class_names = []
        
#     def detect_num_classes(self):
#         """Automatically detect number of classes from dataset structure"""
#         try:
#             # Check if we have train/val structure
#             train_path = os.path.join(self.data_path, 'train')
#             val_path = os.path.join(self.data_path, 'val')
            
#             if os.path.exists(train_path) and os.path.exists(val_path):
#                 # Use train folder for class detection
#                 class_folders = [d for d in os.listdir(train_path) 
#                                if os.path.isdir(os.path.join(train_path, d))]
#                 self.class_names = sorted(class_folders)
#                 detected_classes = len(self.class_names)
                
#                 print(f"Detected train/val structure with {detected_classes} classes: {self.class_names}")
#             else:
#                 # Original structure
#                 class_folders = [d for d in os.listdir(self.data_path) 
#                                if os.path.isdir(os.path.join(self.data_path, d))]
#                 self.class_names = sorted(class_folders)
#                 detected_classes = len(self.class_names)
                
#                 print(f"Detected {detected_classes} classes: {self.class_names}")
            
#             if self.num_classes is None:
#                 self.num_classes = detected_classes
#             elif self.num_classes != detected_classes:
#                 print(f"Warning: Specified {self.num_classes} classes but found {detected_classes}")
#                 self.num_classes = detected_classes
            
#             return self.num_classes
            
#         except Exception as e:
#             print(f"Error detecting classes: {e}")
#             if self.num_classes is None:
#                 self.num_classes = 5  # Default fallback
#             return self.num_classes

#     def create_efficientnet_model(self):
#         """Create EfficientNet-based model"""
#         try:
#             # Detect number of classes first
#             self.detect_num_classes()
            
#             # Use EfficientNetB0 as base model
#             base_model = tf.keras.applications.EfficientNetB0(
#                 input_shape=(*self.img_size, 3),
#                 include_top=False,
#                 weights='imagenet'
#             )
            
#             # Freeze base model initially
#             base_model.trainable = False
            
#             # Add custom classification head
#             model = keras.Sequential([
#                 base_model,
#                 layers.GlobalAveragePooling2D(),
#                 layers.Dropout(0.3),
#                 layers.Dense(128, activation='relu'),
#                 layers.Dropout(0.2),
#                 layers.Dense(self.num_classes, activation='softmax')
#             ])
            
#             return model, True  # Success flag
            
#         except Exception as e:
#             print(f"EfficientNet creation failed: {e}")
#             return None, False
    
#     def create_simple_cnn_model(self):
#         """Create simple CNN model as fallback"""
#         # Detect number of classes first
#         self.detect_num_classes()
        
#         model = keras.Sequential([
#             layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(64, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Conv2D(128, (3, 3), activation='relu'),
#             layers.MaxPooling2D(2, 2),
#             layers.Flatten(),
#             layers.Dropout(0.5),
#             layers.Dense(512, activation='relu'),
#             layers.Dropout(0.3),
#             layers.Dense(self.num_classes, activation='softmax')
#         ])
        
#         return model, True
    
#     def prepare_data(self, batch_size=32):
#         """Prepare training data - handles both split and unsplit datasets"""
#         try:
#             # Detect number of classes first
#             self.detect_num_classes()
            
#             # Check if we have train/val structure
#             train_path = os.path.join(self.data_path, 'train')
#             val_path = os.path.join(self.data_path, 'val')
            
#             if os.path.exists(train_path) and os.path.exists(val_path):
#                 print("Using existing train/val split")
                
#                 # Data augmentation for training
#                 train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255,
#                     rotation_range=20,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     horizontal_flip=True,
#                     zoom_range=0.2
#                 )
                
#                 # Only rescaling for validation
#                 val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255
#                 )
                
#                 # Create training generator
#                 train_generator = train_datagen.flow_from_directory(
#                     train_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical'
#                 )
                
#                 # Create validation generator
#                 val_generator = val_datagen.flow_from_directory(
#                     val_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical'
#                 )
                
#             else:
#                 print("Creating train/val split from data")
                
#                 # Data augmentation for training
#                 train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255,
#                     rotation_range=20,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     horizontal_flip=True,
#                     zoom_range=0.2,
#                     validation_split=0.2
#                 )
                
#                 # Only rescaling for validation
#                 val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255,
#                     validation_split=0.2
#                 )
                
#                 # Create training generator
#                 train_generator = train_datagen.flow_from_directory(
#                     self.data_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical',
#                     subset='training'
#                 )
                
#                 # Create validation generator
#                 val_generator = val_datagen.flow_from_directory(
#                     self.data_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical',
#                     subset='validation'
#                 )
            
#             # Verify the number of classes matches
#             actual_classes = len(train_generator.class_indices)
#             if actual_classes != self.num_classes:
#                 print(f"Updating num_classes from {self.num_classes} to {actual_classes}")
#                 self.num_classes = actual_classes
            
#             print(f"Found {self.num_classes} classes: {list(train_generator.class_indices.keys())}")
#             print(f"Training samples: {train_generator.samples}")
#             print(f"Validation samples: {val_generator.samples}")
            
#             return train_generator, val_generator
            
#         except Exception as e:
#             print(f"Error in data preparation: {e}")
#             return None, None
    
#     def train_model(self, model_type='efficientnet', initial_epochs=30, 
#                    fine_tune_epochs=20, batch_size=32):
#         """Train the model with specified parameters"""
#         print(f"Training configuration:")
#         print(f"Model: {model_type}")
#         print(f"Initial epochs: {initial_epochs}")
#         print(f"Fine-tune epochs: {fine_tune_epochs}")
#         print(f"Batch size: {batch_size}")
#         print("Starting training...")
        
#         try:
#             # Create model based on type
#             if model_type.lower() == 'efficientnet':
#                 self.model, success = self.create_efficientnet_model()
#                 if not success:
#                     print("EfficientNet failed, falling back to Simple CNN")
#                     self.model, success = self.create_simple_cnn_model()
#             else:
#                 self.model, success = self.create_simple_cnn_model()
            
#             if not success:
#                 raise Exception("Failed to create model")
            
#             # Prepare data
#             train_gen, val_gen = self.prepare_data(batch_size)
            
#             if train_gen is None or val_gen is None:
#                 raise Exception("Failed to prepare training data")
            
#             # Compile model
#             self.model.compile(
#                 optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy']
#             )
            
#             # Print model summary
#             print("\nModel Summary:")
#             self.model.summary()
            
#             # Callbacks
#             callbacks = [
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_accuracy',
#                     patience=5,
#                     restore_best_weights=True
#                 ),
#                 keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.2,
#                     patience=3,
#                     min_lr=1e-7
#                 ),
#                 keras.callbacks.ModelCheckpoint(
#                     'best_model.h5',
#                     monitor='val_accuracy',
#                     save_best_only=True,
#                     verbose=1
#                 )
#             ]
            
#             # Initial training
#             print("\nPhase 1: Initial training...")
#             history1 = self.model.fit(
#                 train_gen,
#                 epochs=initial_epochs,
#                 validation_data=val_gen,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             # Fine-tuning (only for EfficientNet)
#             if model_type.lower() == 'efficientnet' and fine_tune_epochs > 0:
#                 print("\nPhase 2: Fine-tuning...")
#                 # Unfreeze some layers for fine-tuning
#                 base_model = self.model.layers[0]
#                 base_model.trainable = True
                
#                 # Fine-tune from this layer onwards
#                 fine_tune_at = 100
#                 for layer in base_model.layers[:fine_tune_at]:
#                     layer.trainable = False
                
#                 # Use lower learning rate for fine-tuning
#                 self.model.compile(
#                     optimizer=keras.optimizers.Adam(1e-5/10),
#                     loss='categorical_crossentropy',
#                     metrics=['accuracy']
#                 )
                
#                 history2 = self.model.fit(
#                     train_gen,
#                     epochs=initial_epochs + fine_tune_epochs,
#                     initial_epoch=len(history1.history['loss']),
#                     validation_data=val_gen,
#                     callbacks=callbacks,
#                     verbose=1
#                 )
            
#             print("\n" + "="*50)
#             print("Training completed successfully!")
#             print("="*50)
#             return True
            
#         except Exception as e:
#             print(f"\nTraining failed with error: {e}")
#             print("Please check your dataset and try again.")
#             if "efficientnet" in model_type.lower():
#                 print("If the error persists, try using the Simple CNN model.")
#             return False
    
#     def save_model(self, filepath='diabetic_retinopathy_model.h5'):
#         """Save the trained model"""
#         if self.model is not None:
#             self.model.save(filepath)
#             print(f"Model saved to {filepath}")
#         else:
#             print("No model to save!")
    
#     def evaluate_model(self, test_data_path=None):
#         """Evaluate the model"""
#         if self.model is None:
#             print("No model to evaluate!")
#             return
        
#         # Use validation data if no test data provided
#         if test_data_path is None:
#             test_data_path = os.path.join(self.data_path, 'val') if os.path.exists(os.path.join(self.data_path, 'val')) else self.data_path
        
#         test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#         test_generator = test_datagen.flow_from_directory(
#             test_data_path,
#             target_size=self.img_size,
#             batch_size=32,
#             class_mode='categorical',
#             shuffle=False
#         )
        
#         # Evaluate
#         print("\nEvaluating model...")
#         loss, accuracy = self.model.evaluate(test_generator, verbose=1)
#         print(f"\nTest Results:")
#         print(f"Test Accuracy: {accuracy:.4f}")
#         print(f"Test Loss: {loss:.4f}")


# def check_dataset_structure(data_path):
#     """Check dataset structure and provide helpful information"""
#     if not os.path.exists(data_path):
#         print(f"Error: Dataset path '{data_path}' does not exist!")
#         return False
    
#     try:
#         # Check if we have train/val structure
#         train_path = os.path.join(data_path, 'train')
#         val_path = os.path.join(data_path, 'val')
        
#         if os.path.exists(train_path) and os.path.exists(val_path):
#             print(f"Dataset structure detected: TRAIN/VAL SPLIT")
#             print(f"Location: {data_path}")
            
#             # Check train folder
#             train_folders = [item for item in os.listdir(train_path) 
#                            if os.path.isdir(os.path.join(train_path, item))]
#             print(f"Training classes: {len(train_folders)}")
            
#             for i, folder in enumerate(sorted(train_folders), 1):
#                 folder_path = os.path.join(train_path, folder)
#                 try:
#                     num_images = len([f for f in os.listdir(folder_path) 
#                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#                     print(f"  {i}. {folder}: {num_images} training images")
#                 except:
#                     print(f"  {i}. {folder}: Unable to count training images")
            
#             # Check val folder
#             val_folders = [item for item in os.listdir(val_path) 
#                          if os.path.isdir(os.path.join(val_path, item))]
#             print(f"Validation classes: {len(val_folders)}")
            
#             for i, folder in enumerate(sorted(val_folders), 1):
#                 folder_path = os.path.join(val_path, folder)
#                 try:
#                     num_images = len([f for f in os.listdir(folder_path) 
#                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#                     print(f"  {i}. {folder}: {num_images} validation images")
#                 except:
#                     print(f"  {i}. {folder}: Unable to count validation images")
            
#         else:
#             # Original single folder structure
#             items = os.listdir(data_path)
#             class_folders = [item for item in items if os.path.isdir(os.path.join(data_path, item))]
            
#             if len(class_folders) == 0:
#                 print(f"Error: No class folders found in '{data_path}'")
#                 return False
            
#             print(f"Dataset structure detected: SINGLE FOLDER")
#             print(f"Location: {data_path}")
#             print(f"Classes found: {len(class_folders)}")
            
#             for i, folder in enumerate(sorted(class_folders), 1):
#                 folder_path = os.path.join(data_path, folder)
#                 try:
#                     num_images = len([f for f in os.listdir(folder_path) 
#                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#                     print(f"  {i}. {folder}: {num_images} images")
#                 except:
#                     print(f"  {i}. {folder}: Unable to count images")
        
#         return True
        
#     except Exception as e:
#         print(f"Error checking dataset: {e}")
#         return False


# def main():
#     """Main training function"""
#     print("Diabetic Retinopathy Detection Training")
#     print("=" * 40)
    
#     # Configuration - UPDATE THIS PATH TO MATCH YOUR STRUCTURE
#     DATA_PATH = "data"  # Your dataset folder
    
#     # Check dataset structure
#     print("Checking dataset structure...")
#     if not check_dataset_structure(DATA_PATH):
#         print("Please fix your dataset structure and try again.")
#         return
    
#     # Get user choice
#     print("\nChoose model type:")
#     print("1. EfficientNet (recommended, but may have compatibility issues)")
#     print("2. Simple CNN (more compatible, faster training)")
    
#     try:
#         choice = input("Enter your choice (1 or 2): ").strip()
        
#         # Set model type based on choice
#         if choice == '1':
#             model_type = 'efficientnet'
#         elif choice == '2':
#             model_type = 'simple_cnn'
#         else:
#             print("Invalid choice, using EfficientNet as default")
#             model_type = 'efficientnet'
        
#         # Initialize trainer (num_classes will be auto-detected) 
#         trainer = DiabetinRetinopathyTrainer(DATA_PATH)
        
#         # Train model
#         success = trainer.train_model(
#             model_type=model_type,
#             initial_epochs=30,
#             fine_tune_epochs=20,
#             batch_size=32
#         )
        
#         if success:
#             # Save the model
#             trainer.save_model('diabetic_retinopathy_model.h5')
            
#             # Evaluate the model
#             trainer.evaluate_model()
#         else:
#             print("\nTraining failed. Please check your dataset and try again.")
#             print("If the error persists, try using the Simple CNN model.")
    
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user.")
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         print("Please check your dataset path and try again.")


# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
# """
# train_model.py - Diabetic Retinopathy Detection Training Script
# Fixed version with proper function definitions and error handling
# """

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# import os
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt

# class DiabetinRetinopathyTrainer:
#     def __init__(self, data_path, img_size=(224, 224), num_classes=None):
#         self.data_path = data_path
#         self.img_size = img_size
#         self.num_classes = num_classes
#         self.model = None
#         self.base_model = None  # Store base model separately for fine-tuning
#         self.class_names = []
        
#     def detect_num_classes(self):
#         """Automatically detect number of classes from dataset structure"""
#         try:
#             # Check if we have train/val structure
#             train_path = os.path.join(self.data_path, 'train')
#             val_path = os.path.join(self.data_path, 'val')
            
#             if os.path.exists(train_path) and os.path.exists(val_path):
#                 # Use train folder for class detection
#                 class_folders = [d for d in os.listdir(train_path) 
#                                if os.path.isdir(os.path.join(train_path, d))]
#                 self.class_names = sorted(class_folders)
#                 detected_classes = len(self.class_names)
                
#                 print(f"Detected train/val structure with {detected_classes} classes: {self.class_names}")
#             else:
#                 # Original structure
#                 class_folders = [d for d in os.listdir(self.data_path) 
#                                if os.path.isdir(os.path.join(self.data_path, d))]
#                 self.class_names = sorted(class_folders)
#                 detected_classes = len(self.class_names)
                
#                 print(f"Detected {detected_classes} classes: {self.class_names}")
            
#             if self.num_classes is None:
#                 self.num_classes = detected_classes
#             elif self.num_classes != detected_classes:
#                 print(f"Warning: Specified {self.num_classes} classes but found {detected_classes}")
#                 self.num_classes = detected_classes
            
#             return self.num_classes
            
#         except Exception as e:
#             print(f"Error detecting classes: {e}")
#             if self.num_classes is None:
#                 self.num_classes = 5  # Default fallback
#             return self.num_classes

#     def create_efficientnet_model(self):
#         """Create EfficientNet-based model"""
#         try:
#             # Detect number of classes first
#             self.detect_num_classes()
            
#             # Use EfficientNetB0 as base model
#             self.base_model = tf.keras.applications.EfficientNetB0(
#                 input_shape=(*self.img_size, 3),
#                 include_top=False,
#                 weights='imagenet'
#             )
            
#             # Freeze base model initially
#             self.base_model.trainable = False
            
#             # Add custom classification head
#             model = keras.Sequential([
#                 self.base_model,
#                 layers.GlobalAveragePooling2D(),
#                 layers.Dropout(0.3),
#                 layers.Dense(128, activation='relu'),
#                 layers.Dropout(0.2),
#                 layers.Dense(self.num_classes, activation='softmax')
#             ])
            
#             return model, True  # Success flag
            
#         except Exception as e:
#             print(f"EfficientNet creation failed: {e}")
#             return None, False
    
#     def create_simple_cnn_model(self):
#         """Create simple CNN model as fallback"""
#         try:
#             # Detect number of classes first
#             self.detect_num_classes()
            
#             model = keras.Sequential([
#                 layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
#                 layers.MaxPooling2D(2, 2),
#                 layers.Conv2D(64, (3, 3), activation='relu'),
#                 layers.MaxPooling2D(2, 2),
#                 layers.Conv2D(128, (3, 3), activation='relu'),
#                 layers.MaxPooling2D(2, 2),
#                 layers.Conv2D(128, (3, 3), activation='relu'),
#                 layers.MaxPooling2D(2, 2),
#                 layers.Flatten(),
#                 layers.Dropout(0.5),
#                 layers.Dense(512, activation='relu'),
#                 layers.Dropout(0.3),
#                 layers.Dense(self.num_classes, activation='softmax')
#             ])
            
#             # For simple CNN, there's no separate base model
#             self.base_model = None
            
#             return model, True
            
#         except Exception as e:
#             print(f"Simple CNN creation failed: {e}")
#             return None, False
    
#     def prepare_data(self, batch_size=32):
#         """Prepare training data - handles both split and unsplit datasets"""
#         try:
#             # Detect number of classes first
#             self.detect_num_classes()
            
#             # Check if we have train/val structure
#             train_path = os.path.join(self.data_path, 'train')
#             val_path = os.path.join(self.data_path, 'val')
            
#             if os.path.exists(train_path) and os.path.exists(val_path):
#                 print("Using existing train/val split")
                
#                 # Data augmentation for training
#                 train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255,
#                     rotation_range=20,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     horizontal_flip=True,
#                     zoom_range=0.2
#                 )
                
#                 # Only rescaling for validation
#                 val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255
#                 )
                
#                 # Create training generator
#                 train_generator = train_datagen.flow_from_directory(
#                     train_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical'
#                 )
                
#                 # Create validation generator
#                 val_generator = val_datagen.flow_from_directory(
#                     val_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical'
#                 )
                
#             else:
#                 print("Creating train/val split from data")
                
#                 # Data augmentation for training
#                 train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255,
#                     rotation_range=20,
#                     width_shift_range=0.2,
#                     height_shift_range=0.2,
#                     horizontal_flip=True,
#                     zoom_range=0.2,
#                     validation_split=0.2
#                 )
                
#                 # Only rescaling for validation
#                 val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                     rescale=1./255,
#                     validation_split=0.2
#                 )
                
#                 # Create training generator
#                 train_generator = train_datagen.flow_from_directory(
#                     self.data_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical',
#                     subset='training'
#                 )
                
#                 # Create validation generator
#                 val_generator = val_datagen.flow_from_directory(
#                     self.data_path,
#                     target_size=self.img_size,
#                     batch_size=batch_size,
#                     class_mode='categorical',
#                     subset='validation'
#                 )
            
#             # Verify the number of classes matches
#             actual_classes = len(train_generator.class_indices)
#             if actual_classes != self.num_classes:
#                 print(f"Updating num_classes from {self.num_classes} to {actual_classes}")
#                 self.num_classes = actual_classes
            
#             print(f"Found {self.num_classes} classes: {list(train_generator.class_indices.keys())}")
#             print(f"Training samples: {train_generator.samples}")
#             print(f"Validation samples: {val_generator.samples}")
            
#             return train_generator, val_generator
            
#         except Exception as e:
#             print(f"Error in data preparation: {e}")
#             return None, None
    
#     def train_model(self, model_type='efficientnet', initial_epochs=30, 
#                    fine_tune_epochs=20, batch_size=32):
#         """Train the model with specified parameters"""
#         print(f"Training configuration:")
#         print(f"Model: {model_type}")
#         print(f"Initial epochs: {initial_epochs}")
#         print(f"Fine-tune epochs: {fine_tune_epochs}")
#         print(f"Batch size: {batch_size}")
#         print("Starting training...")
        
#         try:
#             # Create model based on type
#             if model_type.lower() == 'efficientnet':
#                 self.model, success = self.create_efficientnet_model()
#                 if not success:
#                     print("EfficientNet failed, falling back to Simple CNN")
#                     self.model, success = self.create_simple_cnn_model()
#                     model_type = 'simple_cnn'  # Update model type for later logic
#             else:
#                 self.model, success = self.create_simple_cnn_model()
            
#             if not success or self.model is None:
#                 raise Exception("Failed to create model")
            
#             # Prepare data
#             train_gen, val_gen = self.prepare_data(batch_size)
            
#             if train_gen is None or val_gen is None:
#                 raise Exception("Failed to prepare training data")
            
#             # Compile model
#             self.model.compile(
#                 optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy']
#             )
            
#             # Print model summary
#             print("\nModel Summary:")
#             self.model.summary()
            
#             # Callbacks
#             callbacks = [
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_accuracy',
#                     patience=5,
#                     restore_best_weights=True
#                 ),
#                 keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.2,
#                     patience=3,
#                     min_lr=1e-7
#                 ),
#                 keras.callbacks.ModelCheckpoint(
#                     'best_model.h5',
#                     monitor='val_accuracy',
#                     save_best_only=True,
#                     verbose=1
#                 )
#             ]
            
#             # Initial training
#             print("\nPhase 1: Initial training...")
#             history1 = self.model.fit(
#                 train_gen,
#                 epochs=initial_epochs,
#                 validation_data=val_gen,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             # Fine-tuning (only for EfficientNet and if base_model exists)
#             if (model_type.lower() == 'efficientnet' and 
#                 fine_tune_epochs > 0 and 
#                 self.base_model is not None):
                
#                 print("\nPhase 2: Fine-tuning...")
#                 try:
#                     # Unfreeze the base model for fine-tuning
#                     self.base_model.trainable = True
                    
#                     # Fine-tune from this layer onwards
#                     fine_tune_at = 100
                    
#                     # Make sure we don't exceed the number of layers
#                     if hasattr(self.base_model, 'layers') and len(self.base_model.layers) > fine_tune_at:
#                         for layer in self.base_model.layers[:fine_tune_at]:
#                             layer.trainable = False
#                     else:
#                         # If we can't access layers properly, just use a lower learning rate
#                         print("Using full model fine-tuning with lower learning rate")
                    
#                     # Use lower learning rate for fine-tuning
#                     self.model.compile(
#                         optimizer=keras.optimizers.Adam(1e-5),
#                         loss='categorical_crossentropy',
#                         metrics=['accuracy']
#                     )
                    
#                     history2 = self.model.fit(
#                         train_gen,
#                         epochs=initial_epochs + fine_tune_epochs,
#                         initial_epoch=len(history1.history['loss']),
#                         validation_data=val_gen,
#                         callbacks=callbacks,
#                         verbose=1
#                     )
                    
#                 except Exception as fine_tune_error:
#                     print(f"Fine-tuning failed: {fine_tune_error}")
#                     print("Continuing with initial training results...")
            
#             print("\n" + "="*50)
#             print("Training completed successfully!")
#             print("="*50)
#             return True
            
#         except Exception as e:
#             print(f"\nTraining failed with error: {e}")
#             print("Please check your dataset and try again.")
#             if "efficientnet" in str(model_type).lower():
#                 print("If the error persists, try using the Simple CNN model.")
#             return False
    
#     def save_model(self, filepath='diabetic_retinopathy_model.h5'):
#         """Save the trained model"""
#         try:
#             if self.model is not None:
#                 self.model.save(filepath)
#                 print(f"Model saved to {filepath}")
#             else:
#                 print("No model to save!")
#         except Exception as e:
#             print(f"Error saving model: {e}")
    
#     def evaluate_model(self, test_data_path=None):
#         """Evaluate the model"""
#         if self.model is None:
#             print("No model to evaluate!")
#             return
        
#         try:
#             # Use validation data if no test data provided
#             if test_data_path is None:
#                 test_data_path = os.path.join(self.data_path, 'val') if os.path.exists(os.path.join(self.data_path, 'val')) else self.data_path
            
#             test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#             test_generator = test_datagen.flow_from_directory(
#                 test_data_path,
#                 target_size=self.img_size,
#                 batch_size=32,
#                 class_mode='categorical',
#                 shuffle=False
#             )
            
#             # Evaluate
#             print("\nEvaluating model...")
#             loss, accuracy = self.model.evaluate(test_generator, verbose=1)
#             print(f"\nTest Results:")
#             print(f"Test Accuracy: {accuracy:.4f}")
#             print(f"Test Loss: {loss:.4f}")
            
#         except Exception as e:
#             print(f"Error during evaluation: {e}")


# def check_dataset_structure(data_path):
#     """Check dataset structure and provide helpful information"""
#     if not os.path.exists(data_path):
#         print(f"Error: Dataset path '{data_path}' does not exist!")
#         return False
    
#     try:
#         # Check if we have train/val structure
#         train_path = os.path.join(data_path, 'train')
#         val_path = os.path.join(data_path, 'val')
        
#         if os.path.exists(train_path) and os.path.exists(val_path):
#             print(f"Dataset structure detected: TRAIN/VAL SPLIT")
#             print(f"Location: {data_path}")
            
#             # Check train folder
#             train_folders = [item for item in os.listdir(train_path) 
#                            if os.path.isdir(os.path.join(train_path, item))]
#             print(f"Training classes: {len(train_folders)}")
            
#             for i, folder in enumerate(sorted(train_folders), 1):
#                 folder_path = os.path.join(train_path, folder)
#                 try:
#                     num_images = len([f for f in os.listdir(folder_path) 
#                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#                     print(f"  {i}. {folder}: {num_images} training images")
#                 except:
#                     print(f"  {i}. {folder}: Unable to count training images")
            
#             # Check val folder
#             val_folders = [item for item in os.listdir(val_path) 
#                          if os.path.isdir(os.path.join(val_path, item))]
#             print(f"Validation classes: {len(val_folders)}")
            
#             for i, folder in enumerate(sorted(val_folders), 1):
#                 folder_path = os.path.join(val_path, folder)
#                 try:
#                     num_images = len([f for f in os.listdir(folder_path) 
#                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#                     print(f"  {i}. {folder}: {num_images} validation images")
#                 except:
#                     print(f"  {i}. {folder}: Unable to count validation images")
            
#         else:
#             # Original single folder structure
#             items = os.listdir(data_path)
#             class_folders = [item for item in items if os.path.isdir(os.path.join(data_path, item))]
            
#             if len(class_folders) == 0:
#                 print(f"Error: No class folders found in '{data_path}'")
#                 return False
            
#             print(f"Dataset structure detected: SINGLE FOLDER")
#             print(f"Location: {data_path}")
#             print(f"Classes found: {len(class_folders)}")
            
#             for i, folder in enumerate(sorted(class_folders), 1):
#                 folder_path = os.path.join(data_path, folder)
#                 try:
#                     num_images = len([f for f in os.listdir(folder_path) 
#                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#                     print(f"  {i}. {folder}: {num_images} images")
#                 except:
#                     print(f"  {i}. {folder}: Unable to count images")
        
#         return True
        
#     except Exception as e:
#         print(f"Error checking dataset: {e}")
#         return False


# def main():
#     """Main training function"""
#     print("Diabetic Retinopathy Detection Training")
#     print("=" * 40)
    
#     # Configuration - UPDATE THIS PATH TO MATCH YOUR STRUCTURE
#     DATA_PATH = "data"  # Your dataset folder
    
#     # Check dataset structure
#     print("Checking dataset structure...")
#     if not check_dataset_structure(DATA_PATH):
#         print("Please fix your dataset structure and try again.")
#         return
    
#     # Get user choice
#     print("\nChoose model type:")
#     print("1. EfficientNet (recommended, but may have compatibility issues)")
#     print("2. Simple CNN (more compatible, faster training)")
    
#     try:
#         choice = input("Enter your choice (1 or 2): ").strip()
        
#         # Set model type based on choice
#         if choice == '1':
#             model_type = 'efficientnet'
#         elif choice == '2':
#             model_type = 'simple_cnn'
#         else:
#             print("Invalid choice, using EfficientNet as default")
#             model_type = 'efficientnet'
        
#         # Initialize trainer (num_classes will be auto-detected) 
#         trainer = DiabetinRetinopathyTrainer(DATA_PATH)
        
#         # Train model
#         success = trainer.train_model(
#             model_type=model_type,
#             initial_epochs=30,
#             fine_tune_epochs=20,
#             batch_size=32
#         )
        
#         if success:
#             # Save the model
#             trainer.save_model('diabetic_retinopathy_model.h5')
            
#             # Evaluate the model
#             trainer.evaluate_model()
#         else:
#             print("\nTraining failed. Please check your dataset and try again.")
#             print("If the error persists, try using the Simple CNN model.")
    
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user.")
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         print("Please check your dataset path and try again.")


# if __name__ == "__main__":
#     main()









"""
train_model.py - Diabetic Retinopathy Detection Training Script
Enhanced version with detailed evaluation metrics and visualization
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle

class DiabetinRetinopathyTrainer:
    def __init__(self, data_path, img_size=(224, 224), num_classes=None):
        self.data_path = data_path
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.base_model = None  # Store base model separately for fine-tuning
        self.class_names = []
        
    def detect_num_classes(self):
        """Automatically detect number of classes from dataset structure"""
        try:
            # Check if we have train/val structure
            train_path = os.path.join(self.data_path, 'train')
            val_path = os.path.join(self.data_path, 'val')
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                # Use train folder for class detection
                class_folders = [d for d in os.listdir(train_path) 
                               if os.path.isdir(os.path.join(train_path, d))]
                self.class_names = sorted(class_folders)
                detected_classes = len(self.class_names)
                
                print(f"Detected train/val structure with {detected_classes} classes: {self.class_names}")
            else:
                # Original structure
                class_folders = [d for d in os.listdir(self.data_path) 
                               if os.path.isdir(os.path.join(self.data_path, d))]
                self.class_names = sorted(class_folders)
                detected_classes = len(self.class_names)
                
                print(f"Detected {detected_classes} classes: {self.class_names}")
            
            if self.num_classes is None:
                self.num_classes = detected_classes
            elif self.num_classes != detected_classes:
                print(f"Warning: Specified {self.num_classes} classes but found {detected_classes}")
                self.num_classes = detected_classes
            
            return self.num_classes
            
        except Exception as e:
            print(f"Error detecting classes: {e}")
            if self.num_classes is None:
                self.num_classes = 5  # Default fallback
            return self.num_classes

    def create_efficientnet_model(self):
        """Create EfficientNet-based model"""
        try:
            # Detect number of classes first
            self.detect_num_classes()
            
            # Use EfficientNetB0 as base model
            self.base_model = tf.keras.applications.EfficientNetB0(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model initially
            self.base_model.trainable = False
            
            # Add custom classification head
            model = keras.Sequential([
                self.base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            return model, True  # Success flag
            
        except Exception as e:
            print(f"EfficientNet creation failed: {e}")
            return None, False
    
    def create_simple_cnn_model(self):
        """Create simple CNN model as fallback"""
        try:
            # Detect number of classes first
            self.detect_num_classes()
            
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # For simple CNN, there's no separate base model
            self.base_model = None
            
            return model, True
            
        except Exception as e:
            print(f"Simple CNN creation failed: {e}")
            return None, False
    
    def prepare_data(self, batch_size=32):
        """Prepare training data - handles both split and unsplit datasets"""
        try:
            # Detect number of classes first
            self.detect_num_classes()
            
            # Check if we have train/val structure
            train_path = os.path.join(self.data_path, 'train')
            val_path = os.path.join(self.data_path, 'val')
            
            if os.path.exists(train_path) and os.path.exists(val_path):
                print("Using existing train/val split")
                
                # Data augmentation for training
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    zoom_range=0.2
                )
                
                # Only rescaling for validation
                val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255
                )
                
                # Create training generator
                train_generator = train_datagen.flow_from_directory(
                    train_path,
                    target_size=self.img_size,
                    batch_size=batch_size,
                    class_mode='categorical'
                )
                
                # Create validation generator
                val_generator = val_datagen.flow_from_directory(
                    val_path,
                    target_size=self.img_size,
                    batch_size=batch_size,
                    class_mode='categorical'
                )
                
            else:
                print("Creating train/val split from data")
                
                # Data augmentation for training
                train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    validation_split=0.2
                )
                
                # Only rescaling for validation
                val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                )
                
                # Create training generator
                train_generator = train_datagen.flow_from_directory(
                    self.data_path,
                    target_size=self.img_size,
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='training'
                )
                
                # Create validation generator
                val_generator = val_datagen.flow_from_directory(
                    self.data_path,
                    target_size=self.img_size,
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='validation'
                )
            
            # Verify the number of classes matches
            actual_classes = len(train_generator.class_indices)
            if actual_classes != self.num_classes:
                print(f"Updating num_classes from {self.num_classes} to {actual_classes}")
                self.num_classes = actual_classes
            
            print(f"Found {self.num_classes} classes: {list(train_generator.class_indices.keys())}")
            print(f"Training samples: {train_generator.samples}")
            print(f"Validation samples: {val_generator.samples}")
            
            return train_generator, val_generator
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return None, None
    
    def train_model(self, model_type='efficientnet', initial_epochs=30, 
                   fine_tune_epochs=20, batch_size=32):
        """Train the model with specified parameters"""
        print(f"Training configuration:")
        print(f"Model: {model_type}")
        print(f"Initial epochs: {initial_epochs}")
        print(f"Fine-tune epochs: {fine_tune_epochs}")
        print(f"Batch size: {batch_size}")
        print("Starting training...")
        
        try:
            # Create model based on type
            if model_type.lower() == 'efficientnet':
                self.model, success = self.create_efficientnet_model()
                if not success:
                    print("EfficientNet failed, falling back to Simple CNN")
                    self.model, success = self.create_simple_cnn_model()
                    model_type = 'simple_cnn'  # Update model type for later logic
            else:
                self.model, success = self.create_simple_cnn_model()
            
            if not success or self.model is None:
                raise Exception("Failed to create model")
            
            # Prepare data
            train_gen, val_gen = self.prepare_data(batch_size)
            
            if train_gen is None or val_gen is None:
                raise Exception("Failed to prepare training data")
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Print model summary
            print("\nModel Summary:")
            self.model.summary()
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                ),
                keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Initial training
            print("\nPhase 1: Initial training...")
            history1 = self.model.fit(
                train_gen,
                epochs=initial_epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Fine-tuning (only for EfficientNet and if base_model exists)
            if (model_type.lower() == 'efficientnet' and 
                fine_tune_epochs > 0 and 
                self.base_model is not None):
                
                print("\nPhase 2: Fine-tuning...")
                try:
                    # Unfreeze the base model for fine-tuning
                    self.base_model.trainable = True
                    
                    # Fine-tune from this layer onwards
                    fine_tune_at = 100
                    
                    # Make sure we don't exceed the number of layers
                    if hasattr(self.base_model, 'layers') and len(self.base_model.layers) > fine_tune_at:
                        for layer in self.base_model.layers[:fine_tune_at]:
                            layer.trainable = False
                    else:
                        # If we can't access layers properly, just use a lower learning rate
                        print("Using full model fine-tuning with lower learning rate")
                    
                    # Use lower learning rate for fine-tuning
                    self.model.compile(
                        optimizer=keras.optimizers.Adam(1e-5),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    history2 = self.model.fit(
                        train_gen,
                        epochs=initial_epochs + fine_tune_epochs,
                        initial_epoch=len(history1.history['loss']),
                        validation_data=val_gen,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                except Exception as fine_tune_error:
                    print(f"Fine-tuning failed: {fine_tune_error}")
                    print("Continuing with initial training results...")
            
            print("\n" + "="*50)
            print("Training completed successfully!")
            print("="*50)
            return True
            
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            print("Please check your dataset and try again.")
            if "efficientnet" in str(model_type).lower():
                print("If the error persists, try using the Simple CNN model.")
            return False
    
    def save_model(self, filepath='diabetic_retinopathy_model.h5'):
        """Save the trained model"""
        try:
            if self.model is not None:
                self.model.save(filepath)
                print(f"Model saved to {filepath}")
            else:
                print("No model to save!")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def plot_confusion_matrix(self, y_true, y_pred_classes, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self, y_true, y_pred_proba, class_names):
        """Plot ROC curves for multi-class classification"""
        n_classes = len(class_names)
        
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
        
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def evaluate_model(self, test_data_path=None):
        """Enhanced evaluation with detailed metrics"""
        if self.model is None:
            print("No model to evaluate!")
            return
        
        try:
            # Use validation data if no test data provided
            if test_data_path is None:
                test_data_path = os.path.join(self.data_path, 'val') if os.path.exists(os.path.join(self.data_path, 'val')) else self.data_path
            
            # Create test data generator
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_data_path,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical',
                shuffle=False  # Important: don't shuffle for evaluation
            )
            
            # Get class names from the generator
            class_names = list(test_generator.class_indices.keys())
            
            print("\n" + "="*60)
            print("DETAILED MODEL EVALUATION RESULTS")
            print("="*60)
            
            # Basic evaluation
            print("\n📊 BASIC EVALUATION:")
            print("-" * 30)
            loss, accuracy = self.model.evaluate(test_generator, verbose=1)
            print(f"\n✅ Test Results:")
            print(f"   • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   • Test Loss: {loss:.4f}")
            
            # Get predictions for detailed analysis
            print("\n🔍 GENERATING PREDICTIONS...")
            print("-" * 35)
            
            # Reset generator to ensure proper order
            test_generator.reset()
            
            # Get predictions
            y_pred_proba = self.model.predict(test_generator, verbose=1)
            y_pred_classes = np.argmax(y_pred_proba, axis=1)
            
            # Get true labels
            y_true = test_generator.classes
            
            print(f"✅ Predictions generated:")
            print(f"   • Total samples: {len(y_true)}")
            print(f"   • Classes: {len(class_names)}")
            
            # Classification Report
            print("\n📈 CLASSIFICATION REPORT:")
            print("-" * 40)
            report = classification_report(y_true, y_pred_classes, 
                                         target_names=class_names, 
                                         digits=4)
            print(report)
            
            # Confusion Matrix (text format)
            print("\n🔢 CONFUSION MATRIX:")
            print("-" * 30)
            cm = confusion_matrix(y_true, y_pred_classes)
            print("True\\Predicted", end="")
            for class_name in class_names:
                print(f"\t{class_name[:8]}", end="")
            print()
            
            for i, class_name in enumerate(class_names):
                print(f"{class_name[:12]:<12}", end="")
                for j in range(len(class_names)):
                    print(f"\t{cm[i][j]}", end="")
                print()
            
            # Per-class accuracy
            print("\n🎯 PER-CLASS ACCURACY:")
            print("-" * 35)
            for i, class_name in enumerate(class_names):
                class_mask = (y_true == i)
                if np.sum(class_mask) > 0:
                    class_acc = np.sum((y_pred_classes == i) & class_mask) / np.sum(class_mask)
                    total_samples = np.sum(class_mask)
                    correct_predictions = np.sum((y_pred_classes == i) & class_mask)
                    print(f"   • {class_name:>15}: {class_acc:.4f} ({class_acc*100:.2f}%) - {correct_predictions}/{total_samples}")
            
            # Plot visualizations
            print("\n🎨 GENERATING VISUALIZATIONS...")
            print("-" * 40)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_true, y_pred_classes, class_names)
            print("✅ Confusion matrix saved as 'confusion_matrix.png'")
            
            # Plot ROC curves
            roc_auc = self.plot_roc_curves(y_true, y_pred_proba, class_names)
            print("✅ ROC curves saved as 'roc_curves.png'")
            
            # ROC AUC Summary
            print("\n📊 ROC AUC SCORES:")
            print("-" * 25)
            for i, class_name in enumerate(class_names):
                print(f"   • {class_name:>15}: {roc_auc[i]:.4f}")
            
            avg_auc = np.mean(list(roc_auc.values()))
            print(f"   • {'Average AUC':>15}: {avg_auc:.4f}")
            
            # Summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"📉 Loss: {loss:.4f}")
            print(f"📈 Average ROC AUC: {avg_auc:.4f}")
            print(f"📁 Total Test Samples: {len(y_true)}")
            print(f"🏷️  Number of Classes: {len(class_names)}")
            print(f"💾 Visualizations saved: confusion_matrix.png, roc_curves.png")
            
            if accuracy < 0.5:
                print("\n⚠️  WARNING: Low accuracy detected!")
                print("   Consider:")
                print("   • More training data")
                print("   • Different model architecture")
                print("   • Hyperparameter tuning")
                print("   • Data quality check")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            print(traceback.format_exc())


def check_dataset_structure(data_path):
    """Check dataset structure and provide helpful information"""
    if not os.path.exists(data_path):
        print(f"Error: Dataset path '{data_path}' does not exist!")
        return False
    
    try:
        # Check if we have train/val structure
        train_path = os.path.join(data_path, 'train')
        val_path = os.path.join(data_path, 'val')
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Dataset structure detected: TRAIN/VAL SPLIT")
            print(f"Location: {data_path}")
            
            # Check train folder
            train_folders = [item for item in os.listdir(train_path) 
                           if os.path.isdir(os.path.join(train_path, item))]
            print(f"Training classes: {len(train_folders)}")
            
            for i, folder in enumerate(sorted(train_folders), 1):
                folder_path = os.path.join(train_path, folder)
                try:
                    num_images = len([f for f in os.listdir(folder_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                    print(f"  {i}. {folder}: {num_images} training images")
                except:
                    print(f"  {i}. {folder}: Unable to count training images")
            
            # Check val folder
            val_folders = [item for item in os.listdir(val_path) 
                         if os.path.isdir(os.path.join(val_path, item))]
            print(f"Validation classes: {len(val_folders)}")
            
            for i, folder in enumerate(sorted(val_folders), 1):
                folder_path = os.path.join(val_path, folder)
                try:
                    num_images = len([f for f in os.listdir(folder_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                    print(f"  {i}. {folder}: {num_images} validation images")
                except:
                    print(f"  {i}. {folder}: Unable to count validation images")
            
        else:
            # Original single folder structure
            items = os.listdir(data_path)
            class_folders = [item for item in items if os.path.isdir(os.path.join(data_path, item))]
            
            if len(class_folders) == 0:
                print(f"Error: No class folders found in '{data_path}'")
                return False
            
            print(f"Dataset structure detected: SINGLE FOLDER")
            print(f"Location: {data_path}")
            print(f"Classes found: {len(class_folders)}")
            
            for i, folder in enumerate(sorted(class_folders), 1):
                folder_path = os.path.join(data_path, folder)
                try:
                    num_images = len([f for f in os.listdir(folder_path) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                    print(f"  {i}. {folder}: {num_images} images")
                except:
                    print(f"  {i}. {folder}: Unable to count images")
        
        return True
        
    except Exception as e:
        print(f"Error checking dataset: {e}")
        return False


def main():
    """Main training function"""
    print("Diabetic Retinopathy Detection Training")
    print("=" * 40)
    
    # Configuration - UPDATE THIS PATH TO MATCH YOUR STRUCTURE
    DATA_PATH = "data"  # Your dataset folder
    
    # Check dataset structure
    print("Checking dataset structure...")
    if not check_dataset_structure(DATA_PATH):
        print("Please fix your dataset structure and try again.")
        return
    
    # Get user choice
    print("\nChoose model type:")
    print("1. EfficientNet (recommended, but may have compatibility issues)")
    print("2. Simple CNN (more compatible, faster training)")
    
    try:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        # Set model type based on choice
        if choice == '1':
            model_type = 'efficientnet'
        elif choice == '2':
            model_type = 'simple_cnn'
        else:
            print("Invalid choice, using EfficientNet as default")
            model_type = 'efficientnet'
        
        # Initialize trainer (num_classes will be auto-detected) 
        trainer = DiabetinRetinopathyTrainer(DATA_PATH)
        
        # Train model
        success = trainer.train_model(
            model_type=model_type,
            initial_epochs=30,
            fine_tune_epochs=20,
            batch_size=32
        )
        
        if success:
            # Save the model
            trainer.save_model('diabetic_retinopathy_model.h5')
            
            # Evaluate the model with detailed metrics
            trainer.evaluate_model()
        else:
            print("\nTraining failed. Please check your dataset and try again.")
            print("If the error persists, try using the Simple CNN model.")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your dataset path and try again.")


if __name__ == "__main__":
    main()