import mujoco as mj
import math
from mujoco.glfw import glfw
import numpy as np
from callbacks import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance
import scipy as sc
import my_helpers as helper

break_on_collision = True

xml_path = 'models/mushr_corridor.xml'
view = "third"
assert view in ["first","third"]
simend = 600

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data  = mj.MjData(model)                    # MuJoCo data
cam   = mj.MjvCamera()                        # Abstract camera
opt   = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(800, 600, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

cb = Callbacks(model,data,cam,scene)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, cb.keyboard)
glfw.set_cursor_pos_callback(window, cb.mouse_move)
glfw.set_mouse_button_callback(window, cb.mouse_button)
glfw.set_scroll_callback(window, cb.scroll)

# Example on how to set camera configuration
cam.azimuth = -90 ; cam.elevation = -45 ; cam.distance =  13
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

steer = 0.0
velocity = 0.0
class Controller:
    def __init__(self,model,data):
        data.ctrl[0] = steer
        data.ctrl[1] = velocity
    
    def controller(self, model, data):
        data.ctrl[0] = steer
        data.ctrl[1] = velocity

c = Controller(model,data)
mj.set_mjcb_control(c.controller)

car_obs_list =  [('body', 'buddy'),
            ('body', 'buddy_steering_wheel'),
            ('body', 'buddy_wheel_fl'),
            ('body', 'buddy_wheel_fr'),
            ('body', 'buddy_wheel_bl'),
            ('body', 'buddy_wheel_br')
            ]

human_obs_list = [('site', 'human')]
walls_obs_list = [
                 ('geom', 'obstacle_upper'),
                 ('geom', 'obstacle_lower'),
                 ('geom', 'obstacle_right'),
                 ('geom', 'obstacle_left'),
                 ('geom', 'obstacle_a'),
                 ('geom', 'obstacle_b'),
                 ]

car_obs = helper.filterObs(model, car_obs_list)
human_obs = helper.filterObs(model, human_obs_list)
wall_obs = helper.filterObs(model, walls_obs_list)

wall_rvos = []
for wall in wall_obs:
    shape = []
    center = data.geom_xpos[wall.id][0:2]
    x_rad = wall.size[0]
    y_rad = wall.size[1]
    shape.append(np.add(center, [x_rad, y_rad]))
    shape.append(np.add(center, [x_rad, -y_rad]))
    shape.append(np.add(center, [-x_rad, y_rad]))
    shape.append(np.add(center, [-x_rad, -y_rad]))
    wall_rvos.append([center, np.array([0,0]), shape])

car_goals = [[4, 0, 0], [-4, 0, 0], [-5.0, -5.0, 0]]
current_car_goal = 0
car_finished = False

human_goal = [5.0, 5.0, 0]
last_human_position = [-5.0, -5.0, 0.5]
last_human_trajectory = [0, 0, 0]

goal_velocity = 0.5
goal_thresh = 0.5
rotation_thresh = 0.1

car_state = "Stopped"
stop_time = 100
stop_timer = 0

recalculate_timer = 0
recalculate_time = 10
pause_time_car = 200

trajectory = []
trajectory2 = []
break_flag = False

while not glfw.window_should_close(window) and not break_flag:
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0 and not break_flag):
        mj.mj_step(model,data)

        # Human Values
        human_pos = data.site_xpos[human_obs[0].id]
        if(not np.array_equal(last_human_position, human_pos)):
            last_human_trajectory = human_pos - last_human_position
            last_human_position = np.copy(human_pos)
            
        # Car Values
        car_pos = helper.com(data, car_obs)
        # Hard coded car size estimation
        car_size = [0.25, 0.3, 0]

        car_shape = [
            data.xpos[car_obs[2].id][0:2],
            data.xpos[car_obs[3].id][0:2],
            data.xpos[car_obs[4].id][0:2],
            data.xpos[car_obs[5].id][0:2],
        ]
        human_shape = [
            human_pos[0:2] + [human_obs[0].size[0]/math.sqrt(2), human_obs[0].size[0]/math.sqrt(2)],
            human_pos[0:2] + [-human_obs[0].size[0]/math.sqrt(2), -human_obs[0].size[0]/math.sqrt(2)],
            human_pos[0:2] + [human_obs[0].size[0]/math.sqrt(2), -human_obs[0].size[0]/math.sqrt(2)],
            human_pos[0:2] + [-human_obs[0].size[0]/math.sqrt(2), human_obs[0].size[0]/math.sqrt(2)]
        ]
    
        if car_state == "Stopped":
            if(stop_timer > 0):
                stop_timer -= 1
            else: 
                print("Pursuing Phase")
                car_state = "Pursuing"
                velocity = goal_velocity
                print("\tNavigating to goal", current_car_goal)
        elif car_state == "Finished":
            pass
        elif car_state == "Wait":
            if(recalculate_timer > 0):
                recalculate_timer -= 1
            else: 
                # Steer Calculations (Pure Pursuit and RVO)
                car_rotation = helper.get_car_rotation(data)
                goal_rotation = helper.get_goal_rotation(car_pos, car_goals[current_car_goal])

                will_collide = helper.vo_ab(car_pos[0:2], np.array([0.002*goal_velocity*np.cos(car_rotation), 0.002*goal_velocity*np.sin(car_rotation)]), car_shape, human_pos[0:2], np.array([0,0]), human_shape, thresh=1000, t_iter=5)   
                if(not will_collide):
                    print("No more collision: Pursue")
                    velocity = goal_velocity
                    steer = 0
                    car_state = "Pursuing"
            
                recalculate_timer = pause_time_car
            car_current_goal_finished = distance.euclidean(car_goals[current_car_goal][0:2], car_pos[0:2]) < goal_thresh
            if(car_current_goal_finished):
                print("\tMade it to goal", current_car_goal)
                current_car_goal += 1
                print("Goal Phase")
                car_state = "AtGoal"
        elif car_state == "Pursuing":
            if(recalculate_timer > 0):
                recalculate_timer -= 1
            else:
                # Steer Calculations (Pure Pursuit and RVO)
                car_rotation = helper.get_car_rotation(data)
                goal_rotation = helper.get_goal_rotation(car_pos, car_goals[current_car_goal])

                if(abs(car_rotation - goal_rotation) > rotation_thresh):
                    steer = helper.pure_pursuit(car_rotation, goal_rotation, goal_velocity, Kdd = 0.1)
                else:
                    steer = 0

                will_collide = helper.vo_ab(car_pos[0:2], np.array([0.002*goal_velocity*np.cos(car_rotation), 0.002*goal_velocity*np.sin(car_rotation)]), car_shape, human_pos[0:2], np.array([0,0]), human_shape, thresh=1000, t_iter=10)
                if will_collide:
                    print("Collision with human: Stop and wait")  
                    velocity = 0
                    steer = 0
                    car_state = "Wait" 

                recalculate_timer = recalculate_time
            car_current_goal_finished = distance.euclidean(car_goals[current_car_goal][0:2], car_pos[0:2]) < goal_thresh
            if(car_current_goal_finished):
                print("\tMade it to goal", current_car_goal)
                current_car_goal += 1
                print("Goal Phase")
                car_state = "AtGoal"
        elif car_state == "AtGoal":
            steer = 0
            velocity = 0
            if current_car_goal >= len(car_goals):
                print("Finished Phase")
                car_state = "Finished"
            else:
                print("Stopping Phase")
                stop_timer = stop_time
                car_state = "Stopped"

        # Collisions
        if(break_on_collision):
            if(helper.in_collision_2d(['cylinder', car_pos, car_size], ['cylinder', human_pos, human_obs[0].size])):
                print("Collision between car and human")
                break_flag = True
                break
            for wall in wall_obs:
                wall_pos = data.geom_xpos[wall.id]
                if(helper.in_collision_2d(['rectangle', wall_pos, wall.size], ['cylinder', human_pos, human_obs[0].size])):
                    print("Collision between human and wall")
                    break_flag = True
                    break
                if(helper.in_collision_2d(['rectangle', wall_pos, wall.size], ['cylinder', car_pos, car_size])):
                    print("Collision between car and wall")
                    break_flag = True
                    break

        # All goals reached
        human_finished = distance.euclidean(human_goal[0:2], human_pos[0:2]) < goal_thresh
        car_finished = distance.euclidean(car_goals[-1][0:2], car_pos[0:2]) < goal_thresh
        if(human_finished and car_finished):
            print("Goals Finished")
            break_flag = True
        if(car_finished): velocity = 0

        trajectory.append(np.copy(data.qpos))
        trajectory2.append(np.copy(human_pos))
        if view == "first":
            cam.lookat[0] = data.site_xpos[1][0]
            cam.lookat[1] = data.site_xpos[1][1]
            cam.lookat[2] = data.site_xpos[1][2] + 0.5
            cam.elevation = 0.0
            cam.distance = 1.0
    
    if data.time >= simend:
        break

    # ==================================================================================
    # The below code updates the visualization -- do not modify it!
    # ==================================================================================
    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()
    

glfw.terminate()

import matplotlib.pyplot as plt
# You need the below line only if you're using this inside a Jupyter notebook.
#get_ipython().run_line_magic('matplotlib', 'inline')

trajectory = np.array(trajectory)
trajectory2 = np.array(trajectory2)

plt.figure(figsize=(8,8))
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot(trajectory[:,0],trajectory[:,1],color='black')
plt.plot(trajectory2[:,0],trajectory2[:,1],color='red')
plt.scatter(trajectory[0,0],trajectory[0,1],marker='o', c='blue',label='Start')
plt.scatter(trajectory[-1,0],trajectory[-1,1],marker='x',c='green', label='End')
plt.scatter(trajectory2[0,0],trajectory2[0,1],marker='o', c='blue',)
plt.scatter(trajectory2[-1,0],trajectory2[-1,1],marker='x', c='green',)
plt.legend(loc='best')
plt.show()
