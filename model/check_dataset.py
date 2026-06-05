import os

def check_dataset_structure(data_dir="data"):
    """
    Check and display the current dataset structure
    """
    print(f"Checking dataset structure in: {os.path.abspath(data_dir)}")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' does not exist!")
        return False
    
    # Check main folders
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    print(f"📁 Data directory exists: {data_dir}")
    print(f"📁 Train directory exists: {os.path.exists(train_dir)}")
    print(f"📁 Validation directory exists: {os.path.exists(val_dir)}")
    print()
    
    # Expected class folders
    expected_classes = ['0_no_dr', '1_mild', '2_moderate', '3_severe', '4_proliferative']
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    
    total_train_images = 0
    total_val_images = 0
    
    # Check training folder
    print("🔍 TRAINING FOLDER ANALYSIS:")
    print("-" * 40)
    
    if os.path.exists(train_dir):
        actual_train_folders = os.listdir(train_dir)
        print(f"Found folders in train/: {actual_train_folders}")
        
        for class_folder in expected_classes:
            class_path = os.path.join(train_dir, class_folder)
            if os.path.exists(class_path):
                # Count image files
                files = os.listdir(class_path)
                image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
                non_image_files = [f for f in files if not any(f.lower().endswith(ext) for ext in image_extensions)]
                
                print(f"  ✅ {class_folder}: {len(image_files)} images")
                if len(non_image_files) > 0:
                    print(f"     ⚠️  Non-image files found: {non_image_files[:5]}{'...' if len(non_image_files) > 5 else ''}")
                
                total_train_images += len(image_files)
                
                # Show sample filenames
                if len(image_files) > 0:
                    print(f"     📄 Sample files: {image_files[:3]}{'...' if len(image_files) > 3 else ''}")
            else:
                print(f"  ❌ {class_folder}: folder missing")
    else:
        print("❌ Training directory not found!")
    
    print()
    
    # Check validation folder
    print("🔍 VALIDATION FOLDER ANALYSIS:")
    print("-" * 40)
    
    if os.path.exists(val_dir):
        actual_val_folders = os.listdir(val_dir)
        print(f"Found folders in val/: {actual_val_folders}")
        
        for class_folder in expected_classes:
            class_path = os.path.join(val_dir, class_folder)
            if os.path.exists(class_path):
                files = os.listdir(class_path)
                image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
                non_image_files = [f for f in files if not any(f.lower().endswith(ext) for ext in image_extensions)]
                
                print(f"  ✅ {class_folder}: {len(image_files)} images")
                if len(non_image_files) > 0:
                    print(f"     ⚠️  Non-image files found: {non_image_files[:5]}{'...' if len(non_image_files) > 5 else ''}")
                
                total_val_images += len(image_files)
                
                # Show sample filenames
                if len(image_files) > 0:
                    print(f"     📄 Sample files: {image_files[:3]}{'...' if len(image_files) > 3 else ''}")
            else:
                print(f"  ❌ {class_folder}: folder missing")
    else:
        print("❌ Validation directory not found!")
    
    print()
    print("📊 SUMMARY:")
    print("-" * 20)
    print(f"Total training images: {total_train_images}")
    print(f"Total validation images: {total_val_images}")
    print()
    
    # Recommendations
    if total_train_images == 0 and total_val_images == 0:
        print("🚨 RECOMMENDATIONS:")
        print("1. Check if your images are in the correct folders")
        print("2. Make sure folder names match exactly:")
        for class_name in expected_classes:
            print(f"   - {class_name}/")
        print("3. Verify image file extensions are supported")
        print("4. Run this script to see what's actually in your folders")
    
    return total_train_images > 0 or total_val_images > 0

def create_folder_structure(data_dir="data"):
    """
    Create the required folder structure
    """
    print(f"Creating folder structure in: {os.path.abspath(data_dir)}")
    
    folders_to_create = [
        'data/train/0_no_dr',
        'data/train/1_mild',
        'data/train/2_moderate',
        'data/train/3_severe',
        'data/train/4_proliferative',
        'data/val/0_no_dr',
        'data/val/1_mild',
        'data/val/2_moderate',
        'data/val/3_severe',
        'data/val/4_proliferative'
    ]
    
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    print("\n📁 Folder structure created successfully!")
    print("Now copy your images to the appropriate folders.")

def show_current_directory_contents():
    """
    Show what's in the current directory
    """
    current_dir = os.getcwd()
    print(f"📂 Current working directory: {current_dir}")
    print("📂 Contents of current directory:")
    
    try:
        items = os.listdir(current_dir)
        for item in items:
            if os.path.isdir(item):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except PermissionError:
        print("  ❌ Permission denied to read directory")
    
    print()

def main():
    """Main function"""
    print("🔍 DATASET STRUCTURE CHECKER")
    print("=" * 60)
    
    show_current_directory_contents()
    
    # Check if dataset structure exists
    if not check_dataset_structure():
        print("\n❓ Would you like to create the folder structure? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response in ['y', 'yes']:
                create_folder_structure()
            else:
                print("Please create the folder structure manually and add your images.")
        except:
            print("Creating folder structure...")
            create_folder_structure()
    else:
        print("✅ Dataset structure looks good!")

if __name__ == "__main__":
    main()