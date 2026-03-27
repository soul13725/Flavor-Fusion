import os
import sys

files_to_delete = [
    'E:\\CODING\\Fusion_Flavours_AI\\prompts_updated.py',
    'E:\\CODING\\Fusion_Flavours_AI\\update_prompts.py'
]

for filepath in files_to_delete:
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Deleted: {filepath}")
        else:
            print(f"✗ File not found: {filepath}")
    except Exception as e:
        print(f"✗ Error deleting {filepath}: {e}")

print("\nCleanup complete!")
