import yaml
import matplotlib.pyplot as plt

file_name = \
    '/Users/vladislav.platonov/repo/ColorTracking/ColorTracking/data/ir_ir_5_r/plots/4rot_z.yml'
with open(file_name, 'r') as stream:
    data = yaml.safe_load(stream)
num_data = len(data['frames'])
print(num_data)
step = 2 / (num_data - 1)
current = -step * (num_data - 1) / 2
x = []
y = []
points = data['frames']
for point in range(len(points) - 1):
    point = points[point]
    x.append(current)
    if point['error'] > 1e16:
        y.append(1)
    else:
        y.append(point['error'])
    current += step

plt.plot(x, y)
plt.show()
