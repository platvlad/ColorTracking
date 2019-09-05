import os
from shutil import copy2

input_dir_name = "/Users/vladislav.platonov/CLionProjects/RBOT_experiments/data/RBOT_dataset/"
output_dir_name = "/Users/vladislav.platonov/repo/RBOT2/RBOT/data/RBOT_dataset/"


def write_camera_matrix(output_files):
    camera_file = input_dir_name + "camera_calibration.txt"
    with open(camera_file) as file:
        lines = file.readlines()
        camera_data = lines[1]
        camera_params = camera_data.split('\t')
        fx = float(camera_params[0])
        fy = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        l1 = str([fx, 0.0, -cx, 0.0])
        l2 = str([0.0, fy, -cy, 0.0])
        l3 = str([0.0, 0.0, -1.0, -0.2])
        l4 = str([0.0, 0.0, -1.0, 0.0])
        output_lines = ["matrix: \n",
                        "- " + l1 + "\n",
                        "- " + l2 + "\n",
                        "- " + l3 + "\n",
                        "- " + l4 + "\n",
                        ""]
        for output_file in output_files:
            with open(output_file, 'w') as output:
                output.writelines(output_lines)


def to_int_lists(lst):
    lst = [float(item) for item in lst]
    return lst


def to_opengl_matrix(lst):
    for i in range(len(lst)):
        if 9 > i > 2 or i > 9:
            lst[i] = -lst[i]
    return lst


def write_ground_truth_matrix(output_files):
    ground_truth_file = input_dir_name + "poses_first.txt"
    with open(ground_truth_file) as file:
        lines = file.readlines()[1:]
        for output_file in output_files:
            output_lines = ["frames: \n"]
            for i in range(len(lines)):
                matrix_params = lines[i].split('\t')
                matrix_params = to_int_lists(matrix_params)
                matrix_params = to_opengl_matrix(matrix_params)
                r1 = matrix_params[0:3]
                r2 = matrix_params[3:6]
                r3 = matrix_params[6:9]
                t = matrix_params[9:]
                output_lines.append("- frame: " + str(i + 1) + "\n")
                output_lines.append("  pose: \n")
                output_lines.append("    R: \n")
                output_lines.append("    - " + str(r1) + "\n")
                output_lines.append("    - " + str(r2) + "\n")
                output_lines.append("    - " + str(r3) + "\n")
                output_lines.append("    t: " + str(t) + "\n")
            with open(output_file, 'w') as output:
                output.writelines(output_lines)


def copy_obj(obj_file_name_input, obj_file_name_output):
    lines = []
    with open(obj_file_name_input) as obj_file:
        for line in obj_file:
            if line[0:3] != "vt " and line[0:2] != "f " and line[0:6] != "usemtl" and line[0:6] != "mtllib":
                lines.append(line)
            if line[0:2] == "f ":
                vertices = line.split(' ')[1:]
                v0 = vertices[0].split('/')
                v1 = vertices[1].split('/')
                v2 = vertices[2].split('/')
                line = 'f ' + v0[0] + '//' + v0[2] + ' ' + v1[0] + '//' + v1[2] + ' ' + v2[0] + '//' + v2[2]
                lines.append(line)

    with open(obj_file_name_output, "w") as output_file:
        output_file.writelines(lines)


input_dir = os.fsencode(input_dir_name)
output_dir = os.fsencode(output_dir_name)
output_camera_files = []
output_ground_truth_files = []

for file in os.listdir(input_dir):
    file_name = os.fsdecode(file)
    file_path = os.path.join(input_dir, file)
    if os.path.isdir(file_path):
        output_test_dir = os.path.join(output_dir, file)
        os.mkdir(output_test_dir)

        a_test_dir_path = os.fsencode('a_regular')
        a_test_dir = os.path.join(output_test_dir, a_test_dir_path)
        os.mkdir(a_test_dir)
        b_test_dir_path = os.fsencode('b_dynamic_light')
        b_test_dir = os.path.join(output_test_dir, b_test_dir_path)
        os.mkdir(b_test_dir)
        c_test_dir_path = os.fsencode('c_noisy')
        c_test_dir = os.path.join(output_test_dir, c_test_dir_path)
        os.mkdir(c_test_dir)

        rgb_dir = os.fsencode('rgb')
        a_rgb_dir = os.path.join(a_test_dir, rgb_dir)
        b_rgb_dir = os.path.join(b_test_dir, rgb_dir)
        c_rgb_dir = os.path.join(c_test_dir, rgb_dir)
        os.mkdir(a_rgb_dir)
        os.mkdir(b_rgb_dir)
        os.mkdir(c_rgb_dir)

        frames_dir = os.fsencode('frames')
        frames_dir_path = os.path.join(file_path, frames_dir)
        for png_file in os.listdir(frames_dir_path):
            png_file_name = os.fsdecode(png_file)
            png_file_path = os.path.join(frames_dir_path, png_file)
            if png_file_name.startswith('a_regular'):
                output_png_file_name = png_file_name[len('a_regular'):]
                output_png_file = os.fsencode(output_png_file_name)
                output_png_file_path = os.path.join(a_rgb_dir, output_png_file)
                copy2(png_file_path, output_png_file_path)
            if png_file_name.startswith('b_dynamiclight'):
                output_png_file_name = png_file_name[len('b_dynamiclight'):]
                output_png_file = os.fsencode(output_png_file_name)
                output_png_file_path = os.path.join(b_rgb_dir, output_png_file)
                copy2(png_file_path, output_png_file_path)
            if png_file_name.startswith('c_noisy'):
                output_png_file_name = png_file_name[len('c_noisy'):]
                output_png_file = os.fsencode(output_png_file_name)
                output_png_file_path = os.path.join(c_rgb_dir, output_png_file)
                copy2(png_file_path, output_png_file_path)

        camera_file_name = 'camera.yml'
        camera_file = os.fsencode(camera_file_name)
        a_camera_file = os.path.join(a_test_dir, camera_file)
        b_camera_file = os.path.join(b_test_dir, camera_file)
        c_camera_file = os.path.join(c_test_dir, camera_file)

        ground_truth_file_name = 'ground_truth.yml'
        ground_truth_file = os.fsencode(ground_truth_file_name)
        a_ground_truth_file = os.path.join(a_test_dir, ground_truth_file)
        b_ground_truth_file = os.path.join(b_test_dir, ground_truth_file)
        c_ground_truth_file = os.path.join(c_test_dir, ground_truth_file)

        output_camera_files.append(a_camera_file)
        output_camera_files.append(b_camera_file)
        output_camera_files.append(c_camera_file)
        output_ground_truth_files.append(a_ground_truth_file)
        output_ground_truth_files.append(b_ground_truth_file)
        output_ground_truth_files.append(c_ground_truth_file)

        obj_file_name = file_name + ".obj"
        output_test_dir_name = os.fsdecode(output_test_dir)
        output_obj_file_name = output_test_dir_name + '/mesh.obj'
        input_test_dir_name = os.fsdecode(file_path)
        input_obj_file_name = input_test_dir_name + '/' + obj_file_name
        copy_obj(input_obj_file_name, output_obj_file_name)
        # mtl_file_name = obj_file_name + ".mtl"
        # tex_file_name = file_name + "_tex.png"
        # obj_file = os.fsencode(obj_file_name)
        # output_obj_file = os.fsencode("mesh.obj")
        # mtl_file = os.fsencode(mtl_file_name)
        # tex_file = os.fsencode(tex_file_name)
        # obj_path = os.path.join(file_path, obj_file)
        # mtl_path = os.path.join(file_path, mtl_file)
        # tex_path = os.path.join(file_path, tex_file)
        # obj_output_path = os.path.join(output_test_dir, output_obj_file)
        # mtl_output_path = os.path.join(output_test_dir, mtl_file)
        # tex_output_path = os.path.join(output_test_dir, tex_file)
        # copy2(obj_path, obj_output_path)
        # copy2(mtl_path, mtl_output_path)
        # copy2(tex_path, tex_output_path)


write_camera_matrix(output_camera_files)
write_ground_truth_matrix(output_ground_truth_files)
