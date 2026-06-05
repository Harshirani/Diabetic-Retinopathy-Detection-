# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2

# def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=5):
#     """
#     Create an EfficientNetB0 model for diabetic retinopathy classification.
    
#     Args:
#         input_shape: Input image shape (height, width, channels)
#         num_classes: Number of classes (5 for diabetic retinopathy grades)
    
#     Returns:
#         Compiled Keras model
#     """
    
#     # Load pre-trained EfficientNetB0 without top layers
#     base_model = EfficientNetB0(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
    
#     # Freeze base model layers initially
#     base_model.trainable = False
    
#     # Add custom classification head
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
#     x = Dropout(0.3)(x)
#     predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
#     # Create the model
#     model = Model(inputs=base_model.input, outputs=predictions)
    
#     return model

# def compile_model(model, learning_rate=0.001):
#     """
#     Compile the model with appropriate optimizer and loss function.
    
#     Args:
#         model: Keras model to compile
#         learning_rate: Learning rate for optimizer
    
#     Returns:
#         Compiled model
#     """
    
#     model.compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss='categorical_crossentropy',
#         metrics=['accuracy', 'precision', 'recall']
#     )
    
#     return model

# def unfreeze_model(model, unfreeze_layers=50):
#     """
#     Unfreeze the top layers of the base model for fine-tuning.
    
#     Args:
#         model: Keras model
#         unfreeze_layers: Number of layers to unfreeze from the top
    
#     Returns:
#         Model with unfrozen layers
#     """
    
#     # Get the base model (EfficientNetB0)
#     base_model = model.layers[0]
    
#     # Unfreeze the top layers
#     for layer in base_model.layers[-unfreeze_layers:]:
#         layer.trainable = True
    
#     return model

# def get_model_summary(model):
#     """
#     Get a detailed summary of the model architecture.
    
#     Args:
#         model: Keras model
    
#     Returns:
#         None (prints summary)
#     """
    
#     print("Model Summary:")
#     print("=" * 50)
#     model.summary()
    
#     print("\nTrainable parameters:")
#     trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
#     non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
#     print(f"Trainable: {trainable_params:,}")
#     print(f"Non-trainable: {non_trainable_params:,}")
#     print(f"Total: {trainable_params + non_trainable_params:,}")

# # Class names for diabetic retinopathy
# CLASS_NAMES = [
#     'No DR',
#     'Mild DR',
#     'Moderate DR',
#     'Severe DR',
#     'Proliferative DR'
# ]

# # Image preprocessing function
# def preprocess_image(image_path, target_size=(224, 224)):
#     """
#     Preprocess image for model prediction.
    
#     Args:
#         image_path: Path to the image file
#         target_size: Target size for resizing
    
#     Returns:
#         Preprocessed image array
#     """
    
#     img = tf.keras.preprocessing.image.load_img(
#         image_path, 
#         target_size=target_size
#     )
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create batch axis
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
#     return img_array

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input


def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Create an EfficientNetB0 model for diabetic retinopathy classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of classes (5 for diabetic retinopathy grades)
    
    Returns:
        Compiled Keras model
    """
    
    # Load pre-trained EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_average_pooling')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='dense_1')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01), name='dense_2')(x)
    x = Dropout(0.3, name='dropout_3')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create the model
    #model = Model(inputs=inputs, outputs=predictions, name='diabetic_retinopathy_model')
    model = Model(inputs=base_model.input, outputs=predictions, name='diabetic_retinopathy_model')

    
    print(f"Model created successfully!")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    return model

def create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    """
    Create a simple CNN model as fallback if EfficientNet fails.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of classes
    
    Returns:
        Keras model
    """
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
    
    inputs = Input(shape=input_shape, name='input_layer')
    
    # First convolution block
    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    
    # Second convolution block
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    
    # Third convolution block
    x = Conv2D(128, (3, 3), activation='relu', name='conv3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    
    # Fourth convolution block
    x = Conv2D(256, (3, 3), activation='relu', name='conv4')(x)
    x = BatchNormalization(name='bn4')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)
    
    # Flatten and dense layers
    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01), name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01), name='dense_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    
    # Output layer
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=predictions, name='simple_cnn_model')
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled model
    """
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def unfreeze_model(model, unfreeze_layers=50):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    
    Args:
        model: Keras model
        unfreeze_layers: Number of layers to unfreeze from the top
    
    Returns:
        Model with unfrozen layers
    """
    
    # Get the base model (EfficientNetB0)
    base_model = model.layers[0]
    
    # Unfreeze the top layers
    for layer in base_model.layers[-unfreeze_layers:]:
        layer.trainable = True
    
    return model

def get_model_summary(model):
    """
    Get a detailed summary of the model architecture.
    
    Args:
        model: Keras model
    
    Returns:
        None (prints summary)
    """
    
    print("Model Summary:")
    print("=" * 50)
    model.summary()
    
    print("\nTrainable parameters:")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Trainable: {trainable_params:,}")
    print(f"Non-trainable: {non_trainable_params:,}")
    print(f"Total: {trainable_params + non_trainable_params:,}")

# Class names for diabetic retinopathy
CLASS_NAMES = [
    'No DR',
    'Mild DR',
    'Moderate DR',
    'Severe DR',
    'Proliferative DR'
]

# Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=target_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array