import os
import pandas as pd

folder_path = 'posts_all/'
files = os.listdir(folder_path)

labels = []

for file in files:
    if file.endswith('.txt'):
        with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

            # Check for lines starting with '#' or '//'
            if first_line.startswith('#') or first_line.startswith('//'):
                raw_label = first_line.lstrip('#/ ').strip()
            else:
                raw_label = 'Unknown'

            # 📌 Clean / merge label variants here itself
            raw_label_lower = raw_label.lower()

            if raw_label_lower in ['random msg', 'random message']:
                clean_label = 'random message'
            elif raw_label_lower == 'general message.':
                clean_label = 'general message'
            else:
                clean_label = raw_label

            labels.append({'filename': file, 'label': clean_label})

# Save labels to CSV
df = pd.DataFrame(labels)
df.to_csv('labels.csv', index=False)

print(f"✅ labels.csv created successfully with {len(labels)} entries.")
