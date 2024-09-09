import os

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(u"."):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    for file in files:
        # print(len(path) * '---', file)
        print(root, ' ', file)
        if file in ['llm.pth', 'checkpoint.pth']:
            os.remove(os.path.join(root, file))