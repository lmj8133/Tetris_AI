import os

def process_python_files(directory):
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)

                # Read the content of the file
                with open(file_path, 'r') as file:
                    content = file.read()

                # Print the modified content
                print(f"# Content of {file_path}:")
                print(content)
                print("\n" + "="*50 + "\n")

# Replace 'your_directory_path' with the path of the directory you want to search
directory_path = './'
process_python_files(directory_path)

