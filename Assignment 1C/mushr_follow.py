import mujoco as mj
import my_helpers as helper
from mujoco.glfw import glfw
import numpy as np
import itertools
from callbacks import *
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance

xml_path = 'models/mushr_follow.xml' 
view = "third"
assert view in ["first","third"]
simend = 600

break_on_collision = True

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
        pass
    
    def controller(self,model,data):
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
                 ]
npc_obs_list = [
                ('site', 'npc_1'),
                ('site', 'npc_2'),
                ('site', 'npc_3'),
               ]

car_obs = helper.filterObs(model, car_obs_list)
human_obs = helper.filterObs(model, human_obs_list)
wall_obs = helper.filterObs(model, walls_obs_list)
npc_obs = helper.filterObs(model, npc_obs_list)

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

human_goal = [5, 0, 0]

last_human_pos = np.array([-4, 0, 0.5])
human_vel = np.array([0, 0])
human_finished = False
car_state = "Stopped"

npc1_goals = [[-2.5, 6, 0], [-2.5, -6, 0]]
npc1_cur_goal = 0
npc2_goals = [[0, 6, 0], [0, -6, 0]]
npc2_cur_goal = 0
npc3_goals = [[2.5, 6, 0], [2.5, -6, 0]]
npc3_cur_goal = 0

npc1_path = np.array([0,0])
npc2_path = np.array([0,0])
npc3_path = np.array([0,0])

recalculate_time_npc = 200
recalculate_timer_npc = 0

recalculate_time_car = 10
pause_time_car = 100
recalculate_timer_car = 0

goal_velocity = 0.25
goal_thresh = 0.5
rotation_thresh = 0.1

trajectory_car = []
trajectory_human = []
trajectory_npc1 = []
trajectory_npc2 = []
trajectory_npc3 = []
break_flag = False

car_goals = []
current_car_goal = 0

while not glfw.window_should_close(window) and not break_flag:
    time_prev = data.time

    while (data.time - time_prev < 1.0/60.0 and not break_flag):
        mj.mj_step(model,data)

        # Human Values
        human_pos = data.site_xpos[human_obs[0].id]
        if(not np.array_equal(last_human_pos, human_pos)):
            human_vel = human_pos[0:2] - last_human_pos[0:2]
            last_human_pos = np.copy(human_pos)

        human_shape = [
            human_pos[0:2] + [human_obs[0].size[0]/math.sqrt(2), human_obs[0].size[0]/math.sqrt(2)],
            human_pos[0:2] + [-human_obs[0].size[0]/math.sqrt(2), -human_obs[0].size[0]/math.sqrt(2)],
            human_pos[0:2] + [human_obs[0].size[0]/math.sqrt(2), -human_obs[0].size[0]/math.sqrt(2)],
            human_pos[0:2] + [-human_obs[0].size[0]/math.sqrt(2), human_obs[0].size[0]/math.sqrt(2)]
        ]

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

        # NPC Values
        npc1_pos =  data.site_xpos[npc_obs[0].id]
        npc2_pos =  data.site_xpos[npc_obs[1].id]
        npc3_pos =  data.site_xpos[npc_obs[2].id]

        npc1_shape = [
            npc1_pos[0:2] + [npc_obs[0].size[0]/math.sqrt(2), npc_obs[0].size[0]/math.sqrt(2)],
            npc1_pos[0:2] + [-npc_obs[0].size[0]/math.sqrt(2), -npc_obs[0].size[0]/math.sqrt(2)],
            npc1_pos[0:2] + [npc_obs[0].size[0]/math.sqrt(2), -npc_obs[0].size[0]/math.sqrt(2)],
            npc1_pos[0:2] + [-npc_obs[0].size[0]/math.sqrt(2), npc_obs[0].size[0]/math.sqrt(2)]
        ]
        npc2_shape = [
            npc2_pos[0:2] + [npc_obs[1].size[0]/math.sqrt(2), npc_obs[1].size[0]/math.sqrt(2)],
            npc2_pos[0:2] + [-npc_obs[1].size[0]/math.sqrt(2), -npc_obs[1].size[0]/math.sqrt(2)],
            npc2_pos[0:2] + [npc_obs[1].size[0]/math.sqrt(2), -npc_obs[1].size[0]/math.sqrt(2)],
            npc2_pos[0:2] + [-npc_obs[1].size[0]/math.sqrt(2), npc_obs[1].size[0]/math.sqrt(2)]
        ]
        npc3_shape = [
            npc3_pos[0:2] + [npc_obs[2].size[0]/math.sqrt(2), npc_obs[2].size[0]/math.sqrt(2)],
            npc3_pos[0:2] + [-npc_obs[2].size[0]/math.sqrt(2), -npc_obs[2].size[0]/math.sqrt(2)],
            npc3_pos[0:2] + [npc_obs[2].size[0]/math.sqrt(2), -npc_obs[2].size[0]/math.sqrt(2)],
            npc3_pos[0:2] + [-npc_obs[2].size[0]/math.sqrt(2), npc_obs[2].size[0]/math.sqrt(2)]
        ]
        if car_state == "Stopped":
            human_finished = abs(human_goal[0] - human_pos[0]) < goal_thresh
            if(human_finished):
                print("Human Finished")
                car_goals = np.copy(np.round(trajectory_human, 2))
                car_goals = [k for k, g in itertools.groupby(car_goals, lambda x: (x[0], x[1]))]
                car_goals = car_goals[0::len(car_goals)//20]
                car_state = "Pursuing"
                velocity = goal_velocity
        elif car_state == "Wait":
            if(recalculate_timer_car > 0):
                recalculate_timer_car -= 1
            else: 
                # Steer Calculations (Pure Pursuit and RVO)
                car_rotation = helper.get_car_rotation(data)
                goal_rotation = helper.get_goal_rotation(car_pos, car_goals[current_car_goal])

                will_collide = helper.vo_ab(car_pos[0:2], np.array([0.002*goal_velocity*np.cos(car_rotation), 0.002*goal_velocity*np.sin(car_rotation)]), car_shape, human_pos[0:2], np.array([0,0]), human_shape, thresh=1000, t_iter=5)   
                will_collide = will_collide or helper.vo_ab(car_pos[0:2], np.array([0.002*goal_velocity*np.cos(car_rotation), 0.002*goal_velocity*np.sin(car_rotation)]), car_shape, npc1_pos[0:2], 0.002*npc1_path, npc1_shape, thresh=1000, t_iter=5)
                will_collide = will_collide or helper.vo_ab(car_pos[0:2], np.array([0.002*goal_velocity*np.cos(car_rotation), 0.002*goal_velocity*np.sin(car_rotation)]), car_shape, npc2_pos[0:2], 0.002*npc2_path, npc2_shape, thresh=1000, t_iter=5)
                will_collide = will_collide or helper.vo_ab(car_pos[0:2], np.array([0.002*goal_velocity*np.cos(car_rotation), 0.002*goal_velocity*np.sin(car_rotation)]), car_shape, npc3_pos[0:2], 0.002*npc3_path, npc3_shape, thresh=1000, t_iter=5)
                if(not will_collide):
                    print("No more collision: Pursue")
                    velocity = goal_velocity
                    steer = 0
                    car_state = "Pursuing"
            
                recalculate_timer_car = pause_time_car
            car_current_goal_finished = distance.euclidean(car_goals[current_car_goal][0:2], car_pos[0:2]) < goal_thresh
            if(car_current_goal_finished):
                print("\tMade it to goal", current_car_goal)
                current_car_goal += 1
                print("Goal Phase")
                car_state = "AtGoal"
        elif car_state == "Pursuing": 
            if(recalculate_timer_car > 0):
                recalculate_timer_car -= 1
            else: 
                 # Steer Calculations (Pure Pursuit and RVO)
                car_rotation = helper.get_car_rotation(data)
                goal_rotation = helper.get_goal_rotation(car_pos, car_goals[current_car_goal])

                will_collide = helper.vo_ab(car_pos[0:2], np.array([0.002*velocity*np.cos(car_rotation), 0.002*velocity*np.sin(car_rotation)]), car_shape, human_pos[0:2], np.array([0,0]), human_shape, thresh=1000, t_iter=5)   
                will_collide = will_collide or helper.vo_ab(car_pos[0:2], np.array([0.002*velocity*np.cos(car_rotation), 0.002*velocity*np.sin(car_rotation)]), car_shape, npc1_pos[0:2], 0.002*npc1_path, npc1_shape, thresh=1000, t_iter=5)
                will_collide = will_collide or helper.vo_ab(car_pos[0:2], np.array([0.002*velocity*np.cos(car_rotation), 0.002*velocity*np.sin(car_rotation)]), car_shape, npc2_pos[0:2], 0.002*npc2_path, npc2_shape, thresh=1000, t_iter=5)
                will_collide = will_collide or helper.vo_ab(car_pos[0:2], np.array([0.002*velocity*np.cos(car_rotation), 0.002*velocity*np.sin(car_rotation)]), car_shape, npc3_pos[0:2], 0.002*npc3_path, npc3_shape, thresh=1000, t_iter=5)
                if(will_collide):
                    print("Collision: Stop and Wait")
                    velocity = 0
                    steer = 0
                    car_state = "Wait"

                if(velocity != 0 and abs(car_rotation - goal_rotation) > rotation_thresh):
                    goal_angle = helper.pure_pursuit(car_rotation, goal_rotation, velocity, Kdd = 0.1)  
                else:
                    goal_angle = 0
                
                steer = goal_angle
                recalculate_timer_car = recalculate_time_car
            car_current_goal_finished = distance.euclidean(car_goals[current_car_goal][0:2], car_pos[0:2]) < goal_thresh
            if(car_current_goal_finished):
                print("\tMade it to goal", current_car_goal)
                current_car_goal += 1
                print("Goal Phase")
                car_state = "AtGoal"
        elif car_state == "AtGoal":
            if current_car_goal >= len(car_goals):
                print("Finished Phase")
                velocity = 0
                steer = 0
                car_state = "Finished"
            else:
                car_state = "Pursuing"
        elif car_state == "Finished":
            pass

        # NPC Controls   

        npc_collision_objs = [
            [car_pos[0:2], np.array([0.01*velocity*np.cos(helper.get_car_rotation(data)), 0.01*velocity*np.sin(helper.get_car_rotation(data))]), car_shape],
            [human_pos[0:2], np.array([0,0]), human_shape],  
            [human_pos[0:2], 0.04*human_vel, human_shape]
        ]

        if(recalculate_timer_npc > 0):
            recalculate_timer_npc -= 1
        else: 
            # NPC 1 Movement
            npc1_path = helper.get_npc_pathing(npc1_pos[0:2], npc1_shape, npc_collision_objs + [[npc2_pos[0:2], 0.002*npc2_path, npc2_shape],[npc3_pos[0:2], 0.002*npc3_path, npc3_shape]], npc1_goals[npc1_cur_goal], 0.002)
            # NPC 2 Movement
            npc2_path = helper.get_npc_pathing(npc2_pos[0:2], npc2_shape, npc_collision_objs + [[npc1_pos[0:2], 0.002*npc1_path, npc1_shape],[npc3_pos[0:2], 0.002*npc3_path, npc3_shape]], npc2_goals[npc2_cur_goal], 0.002)
            # NPC 3 Movement
            npc3_path = helper.get_npc_pathing(npc3_pos[0:2], npc3_shape, npc_collision_objs + [[npc2_pos[0:2], 0.002*npc2_path, npc2_shape],[npc1_pos[0:2], 0.002*npc1_path, npc1_shape]], npc3_goals[npc3_cur_goal], 0.002)
            recalculate_timer_npc = recalculate_time_npc

        # At goal, switch goals
        if(distance.euclidean(npc1_pos[0:2], npc1_goals[npc1_cur_goal][0:2]) < goal_thresh):
            npc1_cur_goal = int(not npc1_cur_goal)
            npc1_path = np.array([0,0])
        model.site_pos[npc_obs[0].id][0] += npc1_path[0]
        model.site_pos[npc_obs[0].id][1] += npc1_path[1]
        
        # At goal, switch goals
        if(distance.euclidean(npc2_pos[0:2], npc2_goals[npc2_cur_goal][0:2]) < goal_thresh):
            npc2_cur_goal = int(not npc2_cur_goal)
            npc2_path = np.array([0,0])
        model.site_pos[npc_obs[1].id][0] += npc2_path[0]
        model.site_pos[npc_obs[1].id][1] += npc2_path[1]

        # At goal, switch goals
        if(distance.euclidean(npc3_pos[0:2], npc3_goals[npc3_cur_goal][0:2]) < goal_thresh):
            npc3_cur_goal = int(not npc3_cur_goal)
            npc3_path = np.array([0,0])
        model.site_pos[npc_obs[2].id][0] += npc3_path[0]
        model.site_pos[npc_obs[2].id][1] += npc3_path[1]

        # Collisions
        if(break_on_collision):
            if(helper.in_collision_2d(['cylinder', car_pos, car_size], ['cylinder', human_pos, human_obs[0].size])):
                print("Collision between car and human")
                break_flag = True
                break
            if(helper.in_collision_2d(['cylinder', car_pos, car_size], ['cylinder', npc1_pos, npc_obs[0].size]) or
                helper.in_collision_2d(['cylinder', car_pos, car_size], ['cylinder', npc2_pos, npc_obs[1].size]) or
                helper.in_collision_2d(['cylinder', car_pos, car_size], ['cylinder', npc3_pos, npc_obs[2].size])
                ):
                print("Collision between npc and car")
                break_flag = True
                break
            if(helper.in_collision_2d(['cylinder', human_pos, human_obs[0].size], ['cylinder', npc1_pos, npc_obs[0].size]) or
                helper.in_collision_2d(['cylinder', human_pos, human_obs[0].size], ['cylinder', npc2_pos, npc_obs[1].size]) or
                helper.in_collision_2d(['cylinder', human_pos, human_obs[0].size], ['cylinder', npc3_pos, npc_obs[2].size])
                ):
                print("Collision between npc and human")
                break_flag = True
                break
            if(helper.in_collision_2d(['cylinder', npc2_pos, npc_obs[1].size], ['cylinder', npc1_pos, npc_obs[0].size]) or
                helper.in_collision_2d(['cylinder', npc3_pos, npc_obs[2].size], ['cylinder', npc2_pos, npc_obs[1].size]) or
                helper.in_collision_2d(['cylinder', npc1_pos, npc_obs[0].size], ['cylinder', npc3_pos, npc_obs[2].size])
                ):
                print("Collision between npcs")
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
                if(helper.in_collision_2d(['rectangle', wall_pos, wall.size], ['cylinder', npc1_pos, npc_obs[0].size]) or
                   helper.in_collision_2d(['rectangle', wall_pos, wall.size], ['cylinder', npc2_pos, npc_obs[1].size]) or
                   helper.in_collision_2d(['rectangle', wall_pos, wall.size], ['cylinder', npc3_pos, npc_obs[2].size])
                   ):
                    print("Collision between npc and wall")
                    break_flag = True
                    break

        # All goals reached
        human_finished = human_finished or abs(human_goal[0] - human_pos[0]) < goal_thresh
        car_finished = len(car_goals) != 0 and distance.euclidean(car_goals[-1][0:2], car_pos[0:2]) < goal_thresh
        if(human_finished and car_finished):
            print("Goals Finished")
            break_flag = True

        trajectory_car.append(np.copy(data.qpos))
        trajectory_human.append(np.copy(human_pos))
        trajectory_npc1.append(np.copy(npc1_pos))
        trajectory_npc2.append(np.copy(npc2_pos))
        trajectory_npc3.append(np.copy(npc3_pos))
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

trajectory = np.array(trajectory_human)
trajectory2 = np.array(trajectory_car)
trajectory3 = np.array(trajectory_npc1)
trajectory4 = np.array(trajectory_npc2)
trajectory5 = np.array(trajectory_npc3)

plt.figure(figsize=(8,8))
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot(trajectory[:,0],trajectory[:,1],color='black')
plt.plot(trajectory2[:,0],trajectory2[:,1],color='red')
plt.plot(trajectory3[:,0],trajectory3[:,1],color='blue')
plt.plot(trajectory4[:,0],trajectory4[:,1],color='blue')
plt.plot(trajectory5[:,0],trajectory5[:,1],color='blue')
plt.scatter(trajectory[0,0],trajectory[0,1],marker='o', c='blue',label='Start')
plt.scatter(trajectory[-1,0],trajectory[-1,1],marker='x',c='green', label='End')
plt.scatter(trajectory2[0,0],trajectory2[0,1],marker='o', c='blue',)
plt.scatter(trajectory2[-1,0],trajectory2[-1,1],marker='x', c='green',)
plt.scatter(trajectory3[0,0],trajectory3[0,1], marker='o', c='blue',)
plt.scatter(trajectory4[0,0],trajectory4[0,1],marker='o', c='blue',)
plt.scatter(trajectory5[0,0],trajectory5[0,1],marker='o', c='blue',)
plt.legend(loc='best')
plt.show()