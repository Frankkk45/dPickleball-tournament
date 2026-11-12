import numpy as np
import cv2
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

class RobustTracker:
    def __init__(self):
        self.lower_court = np.array([90, 40, 40])
        self.upper_court = np.array([130, 255, 255])
        
        self.lower_ball = np.array([20, 100, 100])
        self.upper_ball = np.array([40, 255, 255])
        
        self.lower_paddle = np.array([0, 80, 80])
        self.upper_paddle = np.array([20, 255, 255])

    def track(self, obs_raw):
        img = np.transpose(obs_raw, (1, 2, 0))
        img_uint8 = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        mask_court = cv2.inRange(hsv, self.lower_court, self.upper_court)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_court = cv2.morphologyEx(mask_court, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask_court, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None, None, img_bgr
            
        court_contour = max(contours, key=cv2.contourArea)
        court_mask = np.zeros_like(mask_court)
        cv2.drawContours(court_mask, [court_contour], -1, 255, -1)

        cx, cy, cw, ch = cv2.boundingRect(court_contour)
        court_bounds = {
            'y_min': cy,        
            'y_max': cy + ch,   
            'x_min': cx,        
            'x_max': cx + cw 
        }

        img_roi = cv2.bitwise_and(img_bgr, img_bgr, mask=court_mask)
        hsv_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)

        mask_ball = cv2.inRange(hsv_roi, self.lower_ball, self.upper_ball)
        ball_pos = self._get_centroid(mask_ball, min_area=5) 

        mask_paddle = cv2.inRange(hsv_roi, self.lower_paddle, self.upper_paddle)
        mask_paddle = cv2.morphologyEx(mask_paddle, cv2.MORPH_OPEN, kernel, iterations=1)
        
        paddles = self._get_paddles(mask_paddle, min_area=20)
        
        paddle_left = None
        paddle_right = None
        if len(paddles) >= 1:
            h, w = img_bgr.shape[:2]
            mid_x = w / 2
            
            paddles.sort(key=lambda p: p['x'])
            
            paddle_left = paddles[0]
            if len(paddles) > 1:
                paddle_right = paddles[-1]
            else:
                if paddles[0]['x'] < mid_x:
                    paddle_left = paddles[0]
                else:
                    paddle_right = paddles[0]

        return paddle_left, paddle_right, ball_pos, court_bounds, img_bgr

    def _get_centroid(self, mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area: return None
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            return {'x': int(M["m10"] / M["m00"]), 'y': int(M["m01"] / M["m00"])}
        return None

    def _get_paddles(self, mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paddles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else: continue
            
            angle = 0
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                angle = ellipse[2] 
            else:
                rect = cv2.minAreaRect(cnt)
                angle = rect[-1]
                
            paddles.append({'x': cx, 'y': cy, 'angle': angle, 'area': area})
        return paddles