#!/usr/bin/env python3
"""
train_model.py - Diabetic Retinopathy Detection Training Script
Fixed version with proper function definitions and error handling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

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
    
    def evaluate_model(self, test_data_path=None):
        """Evaluate the model"""
        if self.model is None:
            print("No model to evaluate!")
            return
        
        try:
            # Use validation data if no test data provided
            if test_data_path is None:
                test_data_path = os.path.join(self.data_path, 'val') if os.path.exists(os.path.join(self.data_path, 'val')) else self.data_path
            
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_data_path,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )
            
            # Evaluate
            print("\nEvaluating model...")
            loss, accuracy = self.model.evaluate(test_generator, verbose=1)
            print(f"\nTest Results:")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Loss: {loss:.4f}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")


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
            
            # Evaluate the model
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