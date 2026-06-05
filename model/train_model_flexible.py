import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from pathlib import Path
import sys
import glob

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from efficientnet_model import create_efficientnet_model, create_simple_cnn_model
except ImportError:
    print("Warning: Could not import model definitions. Please ensure efficientnet_model.py exists.")
    sys.exit(1)

class FlexibleDRTrainer:
    def __init__(self, data_dir='../data', img_height=224, img_width=224, batch_size=16):
        self.data_dir = os.path.abspath(data_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
        # Auto-detect dataset structure and classes
        self.classes, self.num_classes, self.dataset_type = self._detect_dataset_structure()
        
        print(f"Detected dataset type: {self.dataset_type}")
        print(f"Classes: {self.classes}")
        print(f"Number of classes: {self.num_classes}")
        
    def _detect_dataset_structure(self):
        """Auto-detect the dataset structure and return appropriate class information"""
        train_dir = os.path.join(self.data_dir, 'train')
        
        if not os.path.exists(train_dir):
            print(f"❌ Training directory not found: {train_dir}")
            print("Please ensure your data folder has a 'train' subdirectory")
            sys.exit(1)
        
        # Get all subdirectories in train folder
        subdirs = [d for d in os.listdir(train_dir) 
                  if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')]
        subdirs.sort()
        
        print(f"Found subdirectories in train folder: {subdirs}")
        
        # Check for different dataset structures
        if set(subdirs) == {'DR', 'No_DR'}:
            # Binary classification: DR vs No_DR
            return ['No_DR', 'DR'], 2, 'Binary (DR/No_DR)'
            
        elif set(subdirs) == {'0_no_dr', '1_mild', '2_moderate', '3_severe', '4_proliferative'}:
            # 5-class classification
            return ['0_no_dr', '1_mild', '2_moderate', '3_severe', '4_proliferative'], 5, '5-class (Severity levels)'
            
        elif len(subdirs) > 0:
            # Use whatever folders are found
            return subdirs, len(subdirs), f'Custom ({len(subdirs)} classes)'
            
        else:
            print("❌ No valid class folders found in training directory!")
            print("Expected folders like: DR/No_DR or 0_no_dr/1_mild/etc.")
            sys.exit(1)
    
    def _count_images(self, directory):
        """Count images in a directory"""
        if not os.path.exists(directory):
            return 0
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.gif']
        count = 0
        for ext in image_extensions:
            count += len(glob.glob(os.path.join(directory, ext)))
            count += len(glob.glob(os.path.join(directory, ext.upper())))
        return count
    
    def prepare_data_generators(self):
        """Create data generators based on the detected dataset structure"""
        
        # Check if validation directory exists
        val_dir = os.path.join(self.data_dir, 'val')
        validation_split = None
        
        if os.path.exists(val_dir):
            # Use separate validation directory
            print("Using separate validation directory")
            use_validation_split = False
        else:
            # Use validation split from training data
            print("No separate validation directory found. Using 20% of training data for validation.")
            use_validation_split = True
            validation_split = 0.2
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split if use_validation_split else None
        )
        
        # Training generator
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.classes,
            subset='training' if use_validation_split else None,
            shuffle=True
        )
        
        # Validation generator
        if use_validation_split:
            self.validation_generator = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.classes,
                subset='validation',
                shuffle=True
            )
        else:
            self.validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='categorical',
                classes=self.classes,
                shuffle=True
            )
        
        # Count total images
        train_count = self._count_images_in_generator_dir(os.path.join(self.data_dir, 'train'))
        val_count = self._count_images_in_generator_dir(val_dir) if os.path.exists(val_dir) else int(train_count * 0.2)
        
        print(f"\nDataset Summary:")
        print(f"Training images: {train_count}")
        print(f"Validation images: {val_count}")
        print(f"Total images: {train_count + val_count}")
        
        # Verify generators have data
        if self.train_generator.samples == 0:
            print("❌ ERROR: No training images found!")
            print("Please check your folder structure and image files.")
            sys.exit(1)
        
        if self.validation_generator.samples == 0:
            print("❌ ERROR: No validation images found!")
            sys.exit(1)
        
        print(f"Training generator samples: {self.train_generator.samples}")
        print(f"Validation generator samples: {self.validation_generator.samples}")
        
        return self.train_generator, self.validation_generator
    
    def _count_images_in_generator_dir(self, directory):
        """Count all images in subdirectories"""
        if not os.path.exists(directory):
            return 0
        
        total = 0
        for class_dir in self.classes:
            class_path = os.path.join(directory, class_dir)
            if os.path.exists(class_path):
                total += self._count_images(class_path)
        return total
    
    def create_model(self, model_type='auto'):
        """Create and compile the model"""
        print(f"\nCreating model for {self.num_classes} classes...")
        
        if model_type == 'auto':
            # Try EfficientNet first, fall back to CNN if it fails
            try:
                print("Attempting to create EfficientNet model...")
                self.model = create_efficientnet_model(
                    input_shape=(self.img_height, self.img_width, 3),
                    num_classes=self.num_classes
                )
                print("✅ EfficientNet model created successfully!")
            except Exception as e:
                print(f"⚠️ EfficientNet failed: {str(e)}")
                print("Falling back to Simple CNN model...")
                self.model = create_simple_cnn_model(
                    input_shape=(self.img_height, self.img_width, 3),
                    num_classes=self.num_classes
                )
                print("✅ Simple CNN model created successfully!")
        
        elif model_type == 'efficientnet':
            self.model = create_efficientnet_model(
                input_shape=(self.img_height, self.img_width, 3),
                num_classes=self.num_classes
            )
        
        elif model_type == 'cnn':
            self.model = create_simple_cnn_model(
                input_shape=(self.img_height, self.img_width, 3),
                num_classes=self.num_classes
            )
        
        # Print model summary
        print(f"\nModel Summary:")
        self.model.summary()
        
        return self.model
    
    def train_model(self, initial_epochs=30, fine_tune_epochs=20):
        """Train the model with transfer learning approach"""
        
        if self.model is None:
            print("❌ Model not created. Call create_model() first.")
            return None
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'diabetic_retinopathy_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\n🚀 Starting initial training for {initial_epochs} epochs...")
        
        # Calculate steps
        steps_per_epoch = max(1, self.train_generator.samples // self.batch_size)
        validation_steps = max(1, self.validation_generator.samples // self.batch_size)
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Initial training
        history1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=initial_epochs,
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning (if using EfficientNet)
        if 'efficientnet' in str(type(self.model.layers[1])).lower():
            print(f"\n🔥 Starting fine-tuning for {fine_tune_epochs} epochs...")
            
            # Unfreeze some layers for fine-tuning
            base_model = self.model.layers[1]  # EfficientNet base
            base_model.trainable = True
            
            # Fine-tune from this layer onwards
            fine_tune_at = len(base_model.layers) - 20
            
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Use a lower learning rate for fine-tuning
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Fine-tuning training
            history2 = self.model.fit(
                self.train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=initial_epochs + fine_tune_epochs,
                initial_epoch=initial_epochs,
                validation_data=self.validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            self.history = self._combine_histories(history1, history2)
        else:
            self.history = history1
        
        print("\n✅ Training completed!")
        return self.history
    
    def _combine_histories(self, hist1, hist2):
        """Combine two training histories"""
        combined = {}
        for key in hist1.history.keys():
            combined[key] = hist1.history[key] + hist2.history[key]
        
        # Create a mock history object
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("❌ Model not trained. Train the model first.")
            return None
        
        print("\n📊 Evaluating model on validation data...")
        
        validation_steps = max(1, self.validation_generator.samples // self.batch_size)
        
        # Evaluate on validation data
        val_loss, val_accuracy = self.model.evaluate(
            self.validation_generator,
            steps=validation_steps,
            verbose=1
        )
        
        print(f"\nValidation Results:")
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        return val_loss, val_accuracy
    
    def save_model(self, filepath="diabetic_retinopathy_model.h5"):
        """Save the trained model"""
        if self.model is None:
            print("❌ No model to save.")
            return False
        
        try:
            self.model.save(filepath)
            print(f"✅ Model saved successfully to: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False


def main():
    """Main training function"""
    print("=" * 60)
    print("FLEXIBLE DIABETIC RETINOPATHY DETECTION TRAINER")
    print("=" * 60)
    
    # Check if data directory exists
    data_dir = '../data'
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure your data folder is in the correct location.")
        return
    
    # Configure GPU (if available)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        print(f"✅ GPU detected: {len(physical_devices)} device(s)")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU memory growth enabled")
        except:
            print("Could not enable GPU memory growth")
    else:
        print("⚠️ No GPU detected. Training will use CPU (this will be slower).")
    
    # Initialize trainer
    try:
        trainer = FlexibleDRTrainer(
            data_dir=data_dir,
            img_height=224,
            img_width=224,
            batch_size=16
        )
    except SystemExit:
        return
    
    # Prepare data
    print("\n📁 Preparing data generators...")
    try:
        train_gen, val_gen = trainer.prepare_data_generators()
    except Exception as e:
        print(f"❌ Error preparing data: {str(e)}")
        return
    
    # Create model
    print("\n🤖 Creating model...")
    print("Choose model type:")
    print("1. Auto (try EfficientNet, fallback to CNN)")
    print("2. EfficientNet only")
    print("3. Simple CNN only")
    
    while True:
        try:
            choice = input("Enter choice (1/2/3) [default: 1]: ").strip()
            if choice == "" or choice == "1":
                model_type = "auto"
                break
            elif choice == "2":
                model_type = "efficientnet"
                break
            elif choice == "3":
                model_type = "cnn"
                break
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    try:
        trainer.create_model(model_type=model_type)
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return
    
    # Train model
    print(f"\n🎯 Starting training...")
    print("Training will automatically:")
    print("- Save the best model based on validation accuracy")
    print("- Stop early if no improvement (patience: 10 epochs)")
    print("- Reduce learning rate when validation loss plateaus")
    
    try:
        history = trainer.train_model(initial_epochs=30, fine_tune_epochs=20)
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return
    
    # Evaluate model
    print("\n📈 Final evaluation...")
    try:
        trainer.evaluate_model()
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
    
    # Save model
    print("\n💾 Saving final model...")
    trainer.save_model("diabetic_retinopathy_model.h5")
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("Your model is ready to use!")
    print("Model saved as: diabetic_retinopathy_model.h5")
    print("\nTo use the model:")
    print("1. Run the Flask app: python app.py")
    print("2. Upload retinal images for classification")
    print("3. View predictions and Grad-CAM visualizations")


if __name__ == "__main__":
    main()