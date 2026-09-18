import os

classes = os.listdir("data/MIT_Indoor/train")
classes.sort()

for c in classes:
    print(c)