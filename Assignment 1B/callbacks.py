import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

class Callbacks:
    def __init__(self, model, data, cam, scene):
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0
        
        self.model = model
        self.data = data
        self.cam = cam
        self.scene = scene
    
    def mouse_button(self,window,button,act,mods):
        self.button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        glfw.get_cursor_pos(window)
    
    def mouse_move(self,window,xpos,ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        
        self.lastx = xpos
        self.lasty = ypos
        
        if not (self.button_left or self.button_middle or self.button_right):
            return
        
        w, h = glfw.get_window_size(window)
        
        lshift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        rshift = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        shift  = (lshift or rshift)
        
        if self.button_right:
            if shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        
        mj.mjv_moveCamera(self.model, action, dx/h, dy/h, self.scene, self.cam)
    
    def scroll(self,window,xoffset,yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)
    
    def keyboard(self,window, key, scancode, act, mods):
        angle = np.radians(self.cam.azimuth)
        new_x = np.array([np.cos(angle), np.sin(angle)])
        new_y = np.array([-np.sin(angle), np.cos(angle)])
        
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model,self.data)
            mj.mj_forward(self.model,self.data)
        
        if act == glfw.PRESS and key == glfw.KEY_D:
            self.model.site_pos[1][0] -= 0.05*new_y[0]
            self.model.site_pos[1][1] -= 0.05*new_y[1]
        
        if act == glfw.PRESS and key == glfw.KEY_A:
            self.model.site_pos[1][0] += 0.05*new_y[0]
            self.model.site_pos[1][1] += 0.05*new_y[1]
        
        if act == glfw.PRESS and key == glfw.KEY_S:
            self.model.site_pos[1][0] -= 0.05*new_x[0]
            self.model.site_pos[1][1] -= 0.05*new_x[1]
        
        if act == glfw.PRESS and key == glfw.KEY_W:
            self.model.site_pos[1][0] += 0.05*new_x[0]
            self.model.site_pos[1][1] += 0.05*new_x[1]
        
        if act == glfw.PRESS and key == glfw.KEY_LEFT:
            self.cam.azimuth += 1
        
        if act == glfw.PRESS and key == glfw.KEY_RIGHT:
            self.cam.azimuth -= 1
            
        