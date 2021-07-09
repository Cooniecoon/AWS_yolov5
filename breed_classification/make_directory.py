import os

root_path='./data/Images/'
target_path='./data/crop_images/'
folders=os.listdir(root_path)

for folder in folders:
    name='-'.join(' '.join(folder.split('-')[1:]).split(' '))

    print(name)

    os.makedirs(target_path+name)