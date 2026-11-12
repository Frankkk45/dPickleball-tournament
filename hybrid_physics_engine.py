import numpy as np
from enum import Enum

class State(Enum):
    Q1_FREE_FLIGHT = 1 
    Q2_WALL_COLLISION = 2 
    Q3_PADDLE_CONTACT = 3
    Q4_TELEPORTATION = 4 

class HybridPhysicsPredictor:
    def __init__(self, ball_radius=4.0, contact_threshold=15.0):
        self.r = ball_radius
        self.contact_threshold = contact_threshold
        
        self.y_min = 32.1 
        self.y_max = 62.6 
        self.x_board_min = 44.0
        self.x_board_max = 98.0  
        
        self.current_state = State.Q4_TELEPORTATION
        self.last_ball_pos = None
        self.last_paddle_pos = None
        
        self.accumulated_energy = 0.0 

    def _detect_state_transitions(self, ball, prev_ball, paddles):
        if not ball or not prev_ball:
            return State.Q4_TELEPORTATION
            
        dx = ball['x'] - prev_ball['x']
        dy = ball['y'] - prev_ball['y']
        dist_moved = np.sqrt(dx**2 + dy**2)
        
        if dist_moved > 30.0:
            return State.Q4_TELEPORTATION
            
        for p in paddles:
            if p:
                dist_to_paddle = np.sqrt((ball['x'] - p['x'])**2 + (ball['y'] - p['y'])**2)
                if dist_to_paddle <= self.contact_threshold:
                    self.last_paddle_pos = p
                    return State.Q3_PADDLE_CONTACT
                    
        in_x_bounds = (self.x_board_min - self.r <= ball['x'] <= self.x_board_max + self.r)
        hit_y_bounds = (ball['y'] <= self.y_min + 1.0 or ball['y'] >= self.y_max - 1.0)
        
        if in_x_bounds and hit_y_bounds:
            return State.Q2_WALL_COLLISION
        
        return State.Q1_FREE_FLIGHT

    def step_and_predict(self, ball, prev_ball, paddles, target_x=150):
        new_state = self._detect_state_transitions(ball, prev_ball, paddles)
        
        if self.current_state == State.Q3_PADDLE_CONTACT and new_state == State.Q1_FREE_FLIGHT:
            self.accumulated_energy = 0.0 
            
        self.current_state = new_state
        self.last_ball_pos = ball

        y_predict = 0.5
        t_arrive = 1.0
        is_contact = 0.0
        
        if self.current_state == State.Q4_TELEPORTATION:
            return 0.5, 1.0, 0.0, 0.0

        elif self.current_state == State.Q3_PADDLE_CONTACT:
            is_contact = 1.0
            p_curr = self.last_paddle_pos
            
            vx = p_curr['x'] - self.last_paddle_pos['x'] if hasattr(self, 'prev_p') else 0
            vy = p_curr['y'] - self.last_paddle_pos['y'] if hasattr(self, 'prev_p') else 0
            self.prev_p = p_curr 
            
            self.accumulated_energy += np.sqrt(vx**2 + vy**2)
            
            virtual_prev_ball = {'x': ball['x'] - vx, 'y': ball['y'] - vy}
            y_predict, t_arrive = self._algebraic_fold_predict(ball, virtual_prev_ball, target_x)
            
        else:
            y_predict, t_arrive = self._algebraic_fold_predict(ball, prev_ball, target_x)
            self.accumulated_energy = 0.0

        return y_predict, t_arrive, is_contact, self.accumulated_energy

    def _algebraic_fold_predict(self, pos_curr, pos_prev, target_x):
        if not pos_curr or not pos_prev:
            return 0.5, 1.0
            
        vx = pos_curr['x'] - pos_prev['x']
        vy = pos_curr['y'] - pos_prev['y']
        
        if abs(vx) < 0.1:
            return 0.5, 1.0
            
        dx = target_x - pos_curr['x']
        if (vx > 0 and dx <= 0) or (vx < 0 and dx >= 0):
             return 0.5, 1.0
             
        frames_to_arrive = abs(dx / vx)
        
        y_theoretical = pos_curr['y'] + vy * frames_to_arrive
        
        effective_height = self.y_max - self.y_min
        y_shifted = y_theoretical - self.y_min
        
        num_reflections = int(abs(y_shifted) // effective_height)
        remainder = abs(y_shifted) % effective_height
        
        if y_shifted < 0:
            if num_reflections % 2 == 0:
                y_target_shifted = remainder
            else:
                y_target_shifted = effective_height - remainder
        else:
            if num_reflections % 2 == 0:
                y_target_shifted = remainder
            else:
                y_target_shifted = effective_height - remainder
                
        y_target_real = y_target_shifted + self.y_min
        y_target_norm = np.clip(y_target_real / 84.0, 0.0, 1.0)
        time_left_norm = np.clip(frames_to_arrive / 100.0, 0.0, 1.0)
        
        return y_target_norm, time_left_norm