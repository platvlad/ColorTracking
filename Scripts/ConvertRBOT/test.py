import yaml
import numpy as np

camera_matrix=[[532.80990646, 0.0,342.49522219],
               [0.0,532.93344713,233.88792491],
               [0.0,0.0,1.0]]
dist_coeff = [-2.81325798e-01,2.91150014e-02,1.21234399e-03,-1.40823665e-04,1.54861424e-01]
data = {"camera_matrix": camera_matrix, "dist_coeff": dist_coeff}
with open('output.yml', 'w') as file:
    yaml.dump(data, file)
a = np.array([1, 2, 3])
print(-a)
