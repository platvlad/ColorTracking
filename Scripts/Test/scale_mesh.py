import math
import functools

file_name = "/Users/vladislav.platonov/repo/RBOT2/RBOT/data/RBOT_dataset/cat/mesh.obj"
lines = []
with open(file_name) as obj_file:
    for line in obj_file:
        if line[0:2] == "v ":
            words = line.split()
            for i in range(1, len(words)):
                words[i] = str(0.01 * float(words[i]))
            line = functools.reduce(lambda s1, s2: s1 + " " + s2, words)
            line += '\n'
        lines.append(line)
output_file_name = "/Users/vladislav.platonov/repo/RBOT2/RBOT/data/RBOT_dataset/cat/scaled mesh.obj"
with open(output_file_name, "w") as output_file:
    output_file.writelines(lines)
