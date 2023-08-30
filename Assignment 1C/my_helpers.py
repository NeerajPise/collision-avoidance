
import mujoco as mj
import math
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.spatial import distance

def getActuators(model):
    '''
        Helper function to get all actuators in simulation
    '''
    # model = simulation model
    return([model.actuator(i) for i in range(model.nu)])

def getBodies(model):
    '''
        Helper function to get all bodies in simulation
    '''
    # model = simulation model
    return([model.body(i) for i in range(model.nbody)])

def getGeoms(model):
    '''
        Helper function to get all geoms in simulation
    '''
    # model = simulation model
    return([model.geom(i) for i in range(model.ngeom)])

def getSites(model):
    '''
        Helper function to get all sites in simulation
    '''
    # model = simulation model
    return([model.site(i) for i in range(model.nsite)])

def filterObs(model, filter_list):
    '''
        Helper function to get relevant objects from simulation
    '''
    # model = sumulation model
    # filter_list = list of tuples representing objects to get of form [(type, name)]
    # type = 'site', 'geom', or 'body'
    # name = string for name in xml file
    bodies = getBodies(model)
    geoms = getGeoms(model)
    sites = getSites(model)
    actuators = getActuators(model)

    filtered = []
    for filter in filter_list:
        if filter[0] == 'body':
            for body in bodies:
                if(body.name == filter[1]):
                    filtered.append(body)
        elif filter[0] == 'geom':
            for geom in geoms:
                if(geom.name == filter[1]):
                    filtered.append(geom)
        elif filter[0] == 'site':
            for site in sites:
                if(site.name == filter[1]):
                    filtered.append(site)
        elif filter[0] == 'actuator':
            for actuator in actuators:
                if(actuator.name == filter[1]):
                    filtered.append(actuator)
    return filtered

def com(data, objs):
    '''
        Helper function to get center of mass of set of objects
    '''
    # Data is simulation data
    # objs is list of object views provided in filterObs
    totalmass = 0
    center_x = 0
    center_y = 0
    center_z = 0
    for obj in objs:
        if(obj.mass != None):
            totalmass += obj.mass[0]
            if(isinstance(obj, mj._structs._MjModelBodyViews)):
                center_x += data.xpos[obj.id][0]*obj.mass[0]
                center_y += data.xpos[obj.id][1]*obj.mass[0]
                center_z += data.xpos[obj.id][2]*obj.mass[0]
            elif(isinstance(obj, mj._structs._MjModelGeomViews)):
                center_x += data.geom_xpos[obj.id][0]*obj.mass[0]
                center_y += data.geom_xpos[obj.id][1]*obj.mass[0]
                center_z += data.geom_xpos[obj.id][2]*obj.mass[0]
            else:
                print("ERROR: Unimplemented com for type ", type(obj))
                return None
    center_x /= totalmass
    center_y /= totalmass
    center_z /= totalmass

    return [center_x, center_y, center_z]

def in_collision_2d(obj1, obj2, leniency = 0.05):
    '''
        Helper function to tell if two objects are colliding in 2D space
    '''
    # 2D = doesn't account for z position
    # inputs are of form [type, xpos, size]
    # type is 'rectangle' or 'cylinder'
    # xpos is current position [x,y,z]
    # size is size array
    if(obj1[0] == 'rectangle' and obj2[0] == 'rectangle'):
        # Two rectangles case
        # Assuming no rotation and same orientation (x lines are parallel and y lines are parallel)
        # Calculate the y distance and x distance of two points
        # Compare to the sum of their x and y radii
        x_dist = abs(obj1[1][0] - obj2[1][0])
        y_dist = abs(obj1[1][1] - obj2[1][1])
        x_thresh = obj1[2][0] + obj2[2][0] - leniency/2
        y_thresh = obj1[2][1] + obj2[2][1] - leniency/2
        if(x_dist < x_thresh and y_dist < y_thresh):
            return True
    elif(obj1[0] == 'rectangle' or obj2[0] == 'rectangle'):
        # One rectangle and one cylinder
        # Simply treat the cylinder as if it were a square of length 2*radius
        # (This may result in some inaccuracy around corners)
        # Treat same as rectangle rectangle case
        x_dist = abs(obj1[1][0] - obj2[1][0])
        y_dist = abs(obj1[1][1] - obj2[1][1])
        if(obj1[0] == 'rectangle'):
            x_thresh = obj1[2][0] + obj2[2][0] - leniency/2
            y_thresh = obj1[2][1] + obj2[2][0] - leniency/2
        else:
            x_thresh = obj1[2][0] + obj2[2][0] - leniency/2
            y_thresh = obj1[2][0] + obj2[2][1] - leniency/2
        if(x_dist < x_thresh and y_dist < y_thresh):
            return True
    else:
        # Two cylinders case 
        # if the euclidian distance is less than the sum of the radii then we have collision
        euclid = math.sqrt((obj1[1][0] - obj2[1][0])**2 + (obj1[1][1] - obj2[1][1])**2)
        threshold = obj1[2][0] + obj2[2][0] - leniency
        if(euclid < threshold):
            return True
    return False

def get_car_rotation(data):
    '''
    Helper function to get the cars angle of rotation in space along the Z axis.
    '''
    # Data is simulation data
    car_rotation = R.from_quat(np.append(data.qpos[4:7],data.qpos[3])).as_euler('xyz', degrees=False)[2]
    if(car_rotation < 0):
        car_rotation = car_rotation + (2*math.pi)*(math.ceil(abs(car_rotation)/(2*math.pi)))
    car_rotation = np.round(car_rotation, 3)
    return car_rotation

def unit_vec_to_rad(vec):
    '''
    Helper to convert an x,y unit vector to its rotation about the z axis in radians
    '''
    # vec is the unit vector 
    rad = math.atan(vec[1]/vec[0])
    if(rad < 0):
        rad = rad + (2*math.pi)*(math.ceil(abs(rad)/(2*math.pi)))
    if(vec[1] <= 0 and vec[0] <= 0):
        rad += math.pi
    elif(vec[0] <= 0):
        rad -= math.pi
    return rad

def get_goal_rotation(car_pos, goal_pos):
    '''
    Helper function to get the desired angle of rotation along the z axis for the car to reach the goal
    '''
    # Car_pos is the cars coordinates at that point in time
    # goal_pos is the goal coordinates
    if(car_pos[0] - goal_pos[0] == 0):
        goal_rotation = math.pi/2 if goal_pos[1] - car_pos[1] > 0 else 3*math.pi/2
    else:
        y_vec = (goal_pos[1] - car_pos[1])
        x_vec = (goal_pos[0] - car_pos[0])
        goal_rotation = np.round(unit_vec_to_rad([x_vec, y_vec]), 3)
    return goal_rotation

def pure_pursuit(car_rotation, goal_rotation, car_vel, wheel_base = 0.3, Kdd = 10.5):
    '''
    Helper function to perform the calculations for pure pursuit to calculate
    the angle of steering to reach the goal
    '''
    # Car_rotation is the angle of rotation for the car about the z axis
    # Goal_rotation is the desired angle of rotation for the car about the z axis
    # car_vel is the speed of the car
    # wheel_base is the length of the car's wheel base
    # Kdd is the adjustable constant in pure pursuit calculation
    alpha = goal_rotation - car_rotation
    return round(np.arctan((2*wheel_base*np.sin(alpha))/(Kdd*car_vel)),3)

def get_center(A):
    '''
    Helper to get center point of a rectangle given its corner coordinates
    '''
    # A is an array of points (x,y) representing the shape
    A = np.array(A)
    max_x = np.max(A[:,0])
    max_y = np.max(A[:,1])
    min_x = np.min(A[:,0])
    min_y = np.min(A[:,1])
    return [(max_x+min_x)/2, (max_y+min_y)/2]

def center_offset_points(A):
    '''
    Helper to get the offset of all points about their center.
    E.g. If you have square (2, 2), (2, 0), (0, 0), (0, 2)
    then its center is (1,1) and the offset is (1, 1), (1, -1), (-1, -1), (-1, 1)
    '''
    # A is an array of points (x,y) representing the shape
    center = get_center(A)
    offset_points = []
    for point in A:
        offset_points.append(np.array(point)-np.array(center))
    return offset_points

def center_around(A, p):
    '''
    Given a shape and point, get a new set of points if the given 
    shape were centered around point p
    '''
    # A is an array of points (x,y) representing the shape
    # p is the point to center around
    shape = []
    offset_points = center_offset_points(A)
    for point in offset_points:
        shape.append(point+p)
    return shape

def minkowski_sum(A, B):
    '''
    Calculate the minkowski sum of shapes A and B
    Used for calculating VO
    '''
    # A is an array of points (x,y) representing the shape
    # B is an array of points (x,y) representing the shape
    minkowski_sum = []
    for a_vec in A:
        for b_vec in B:
            minkowski_sum.append(a_vec + b_vec)
    return minkowski_sum

def vo_lambda_linear(p, v, t_thresh, t_iter=1):
    '''
    Calculate the trajectory of an object at point p moving at velocity v
    at all points in time up until t_thresh, iterating by t_iter
    '''
    # p is the starting point
    # v is the velocity vector
    # t_thresh is the max time to calculate up till
    # t_iter is the iteration step between time points
    lambda_set = []
    for t in range(0, t_thresh, t_iter):
        lambda_set.append(p+v*t)
    return lambda_set

def quadrilateralize(A):
    '''
    Takes 2D points and converts them to a quadrilateral
    based on max/min x/y values
    '''
    # A is an array of points (x,y) representing the shape
    A = np.array(A)
    max_x = np.max(A[:,0])
    max_y = np.max(A[:,1])
    min_x = np.min(A[:,0])
    min_y = np.min(A[:,1])
    return [min_x, max_x, min_y, max_y]

def vo_ab(pa, va, A, pb, vb, B, thresh=10, t_iter=1, sensitivity=0.25):
    '''
    Returns if va is inside the velocity obstacle of B.
    '''
    # pa is the position of A
    # va is the velocity vector of A
    # A is an array of points (x,y) representing the shape
    # pb is the position of B
    # vb is the velocity vector of B
    # B is an array of points (x,y) representing the shape
    # thresh is the max time value for vo_lambda
    # t_iter is the iteration step for time in vo_lambda
    # sensitivity is the padding given to the shape defined by the minkowski sum 
    # Assumes all shapes to be quadrilateral
    minkowski = np.round(quadrilateralize(center_around(minkowski_sum(np.array(B), -np.array(A)), pb)), 3)
    l = np.round(vo_lambda_linear(np.array(pa), np.array(va)-np.array(vb), t_thresh=thresh, t_iter=t_iter), 3)
    for i, pos in enumerate(l):
        if(minkowski[0] - sensitivity < pos[0] and 
           pos[0] < minkowski[1] + sensitivity and
           minkowski[2] - sensitivity < pos[1] and 
           pos[1] < minkowski[3] + sensitivity):
            return True
    return False

def rotate_unit_vec(vel, rotate_factor):
    '''
    Helper to rotate a unit vector by an angle
    '''
    # vel is unit vector to rotate
    # rotate_factor is angle to rotate by in radians
    new_vel = [None, None]
    new_vel[0] = np.cos(rotate_factor)*vel[0] - np.sin(rotate_factor)*vel[1]
    new_vel[1] = np.sin(rotate_factor)*vel[0] + np.cos(rotate_factor)*vel[1]
    return np.array(new_vel)
    
def get_npc_pathing(npc_pos, A_npc, VOs, goal_pos, step_dist, t_thresh=1000, t_iter=5, rotation_inc = math.pi/12):
    '''
    Calculate the movement vector for an NPC 
    The vector must: 
    avoids all obstacles
    try to be as close as possible to the direct vector between the npc and its goal
    '''
    # NPC_pos is coordinates of npc
    # A_npc is the shape of the npc
    # VOs is an array of velocity obstacle variables in the form [pos, vel, shape]
    # Where pos is that objects position, vel is its current velocity vector, and shape is an array of points representing its shape
    # goal_pos is the position of the NPCs current goal
    # step_dist is the amount the npc should move per simulation step
    # t_thresh is the max time for collision calculations VO
    # t_iter is the iteration step for time in VO
    # rotation_inc is the amount to rotate the direction vector by when trying to search for a movement vector that avoids collisions
    npc_angle = get_goal_rotation(npc_pos, goal_pos)
    npc_vec = np.array([np.cos(npc_angle), np.sin(npc_angle)])
    
    will_collide = False
    for vo in VOs:
        will_collide = will_collide or vo_ab(npc_pos, step_dist*npc_vec, A_npc, vo[0], vo[1], vo[2], thresh=t_thresh, t_iter=t_iter)
    if(will_collide):
        cw = np.copy(npc_vec)
        ccw = np.copy(npc_vec)
        cw_collide = True
        ccw_collide = True
        count = 0
        while(cw_collide and ccw_collide and count < int((2*math.pi)/rotation_inc)):
            cw_collide = False
            ccw_collide = False
            cw = rotate_unit_vec(cw, -rotation_inc)
            ccw = rotate_unit_vec(ccw, rotation_inc)
            for vo in VOs:
                cw_collide = cw_collide or vo_ab(npc_pos, step_dist*cw, A_npc, vo[0], vo[1], vo[2], thresh=t_thresh, t_iter=t_iter)
                ccw_collide = ccw_collide or vo_ab(npc_pos, step_dist*ccw, A_npc, vo[0], vo[1], vo[2], thresh=t_thresh, t_iter=t_iter)
            count += 1
        if(count >= int((2*math.pi)/rotation_inc)):
            return np.array([0,0])
        if(not cw_collide):
            npc_vec = cw
        else:
            npc_vec = ccw

    return np.array([npc_vec[0]*step_dist, npc_vec[1]*step_dist])