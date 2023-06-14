import fractions
import glob
import os
import json
import glob
import shutil

root_dir = "./spiderman"
save_root_dir = "./spiderman_new"

# original dirs
image_dir = os.path.join(root_dir, "test")
json_dir = os.path.join(root_dir, "transforms_test.json")
keypoint_dir = os.path.join(root_dir, "keypoint")
src_dir = os.path.join(root_dir, "src_id.txt")
# new dirs
os.makedirs(save_root_dir, exist_ok=True)
image_save_dir = os.path.join(save_root_dir, "test")
os.makedirs(image_save_dir, exist_ok=True)
json_save_dir = os.path.join(save_root_dir, "transforms_test.json")
keypoint_save_dir = os.path.join(save_root_dir, "keypoint")
os.makedirs(keypoint_save_dir, exist_ok=True)
src_save_dir = os.path.join(save_root_dir, "src_id.txt")

# change json names
with open(json_dir) as f:
    json_file = json.load(f)
frames = json_file["frames"]
for i in range(len(frames)):
    file_path = frames[i]["file_path"].split("/")
    frame_id = int(file_path[1])
    if i == 0:
        first_id = frame_id
    frame_name = file_path[0] + "/{:04d}".format(frame_id - first_id)
    json_file["frames"][i]["file_path"] = frame_name
with open(json_save_dir, 'w') as f:
    json.dump(json_file, f, indent=4)

# change image names
for i, imagename in enumerate(os.listdir(image_dir)):
    if imagename.endswith(".png") or imagename.endswith(".jpg"):
        print(imagename)
        file_path = imagename.split(".")
        image_id = int(file_path[0])
        print(image_id - first_id)
        shutil.copy(os.path.join(image_dir, imagename), os.path.join(image_save_dir, "{:04d}".format(image_id - first_id) + '.' + file_path[1]))

# change keypoint name 
for i, objname in enumerate(os.listdir(keypoint_dir)):
    if objname.endswith(".mtl") or objname.endswith(".obj"):
        file_path = objname.split(".")
        obj_id = int(file_path[0])
        shutil.copy(os.path.join(keypoint_dir, objname), os.path.join(keypoint_save_dir, str(obj_id - first_id) + '.' + file_path[1]))

# change src_id name
with open(src_dir, 'r') as f:
    src_id = int(f.readlines()[0])
with open(src_save_dir, 'w') as f:
    f.write(str(src_id - first_id))

# move train files to new dir