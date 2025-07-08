import os

def update_master_classes(data_dir, master_file='src/classes/master_classes.txt'):

    """Update the master classes file with new classes found in the specified data directory."""
    
    # Read existing classes
    if os.path.exists(master_file):
        with open(master_file, 'r') as f:
            master_classes = set(line.strip() for line in f)
    else:
        # master_classes = set()
        pass
    print(os.getcwd())
    # print(f"Current master classes: {len(master_classes)}")
    print('orignal master classes',master_classes if master_classes else 'None')

    # Find all class folders in new data
    new_classes = set(os.listdir(data_dir))
    # Update master list
    print('new classes', new_classes)
    updated = master_classes | new_classes
    with open(master_file, 'w') as f:
        for cls in sorted(updated):
            f.write(f"{cls}\n")
    print(f"Updated master_classes.txt with {len(updated)} classes.")

if __name__ == "__main__":
    # Example usage
    data_dir = 'Data/Images of 51 Fruits and Vegetables/Train'
    update_master_classes(data_dir)
