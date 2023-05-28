import fractions
import glob
import os
import json
import glob
import shutil

root_dir = "./human"
save_root_dir = "./human_crop"
start_id = 65
end_id = 115

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

# load json
with open(json_dir) as f:
    json_file = json.load(f)
# create new json
json_write_file = json_file.copy()
json_write_file["frames"] = []
# load src_id
with open(src_dir, 'r') as f:
    src_id = int(f.readlines()[0])
for i in range(end_id - start_id):
    # change json and copy
    json_i = json_file["frames"][start_id + i]
    file_path = json_i["file_path"].split("/")
    frame_id = int(file_path[1])
    frame_name = file_path[0] + "/{:04d}".format(frame_id - start_id)
    json_i["file_path"] = frame_name
    json_write_file["frames"].append(json_i)
    # change image name and copy
    shutil.copy(os.path.join(image_dir, "{:04d}".format(start_id + i) + ".png"), os.path.join(image_save_dir, "{:04d}".format(frame_id - start_id)+ ".png"))
    # change keypoint name and copy
    shutil.copy(os.path.join(keypoint_dir, str(start_id + i) + ".obj"), os.path.join(keypoint_save_dir, str(frame_id - start_id)+ ".obj"))
    shutil.copy(os.path.join(keypoint_dir, str(start_id + i) + ".mtl"), os.path.join(keypoint_save_dir, str(frame_id - start_id)+ ".mtl"))
    # change src_id
    if src_id == frame_id:
        with open(src_save_dir, 'w') as f:
            f.write(str(src_id - start_id))
    
# write to json   
with open(json_save_dir, 'w') as f:
    json.dump(json_write_file, f, indent=4)
