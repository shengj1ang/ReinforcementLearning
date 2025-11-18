import numpy as np

iteration_res=[
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
]
reward_matrix=[
    [0,-1,-1,-1],
    [-1,-1,-1,-1],
    [-1,-1,-1,-1],
    [-1,-1,-1,0]
]
r=1

def get_value_up(x,y):
    global last_iteration
    if y-1<0: # Hit the wall
        return last_iteration[x][y]
    else:
        return last_iteration[x][y-1]

def get_value_down(x,y):
    global last_iteration
    if y+1>3: # Hit the wall
        return last_iteration[x][y]
    else:
        return last_iteration[x][y+1]


def get_value_left(x,y):
    global last_iteration
    if x-1<0: # Hit the wall
        return last_iteration[x][y]
    else:
        return last_iteration[x-1][y]

def get_value_right(x,y):
    global last_iteration
    if x+1>3: # Hit the wall
        return last_iteration[x][y]
    else:
        return last_iteration[x+1][y]

def pharse_matrix(matrix):
    print(np.array(matrix))
for k in range(0,3):
    print(f"Interation: {k}")
    last_iteration = iteration_res
    print(last_iteration)
    for x in range(0,4):
        for y in range(0,4):
            
            value_of_this_state=reward_matrix[x][y]+r*0.25*(get_value_up(x,y)+get_value_down(x,y)+get_value_left(x,y)+get_value_right(x,y))
            print({
                "x":x,
                "y":y,
                "up":get_value_up(x,y),
                "down":get_value_down(x,y),
                "left":get_value_left(x,y),
                "right":get_value_right(x,y),
                "reward_matrix":reward_matrix[x][y],
                "value_of_this_state":value_of_this_state
                })
            iteration_res[x][y]=value_of_this_state
    pharse_matrix(iteration_res)