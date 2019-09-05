file_name = "/Users/vladislav.platonov/repo/RBOT2/RBOT/data/RBOT_dataset/cat/mesh.obj"
lines = []
with open(file_name) as obj_file:
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


output_file_name = "/Users/vladislav.platonov/repo/RBOT2/RBOT/data/RBOT_dataset/cat/mesh_no_texture.obj"
with open(output_file_name, "w") as output_file:
        output_file.writelines(lines)
