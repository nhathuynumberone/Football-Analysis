import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distant

class PlayerBallAssigner:
    def __init__(self) -> None:
        self.max_player_ball_distant = 70
        
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        
        min_distant = 999999999
        assigned_player = -1
        
        for player_id, player in players.items():
            distant_left = measure_distant((player['bbox'][0], player['bbox'][-1]), ball_position)
            distant_right = measure_distant((player['bbox'][2], player['bbox'][-1]), ball_position)
            
            distant = min(distant_left, distant_right)
            
            if distant < self.max_player_ball_distant:
                if distant < min_distant:
                    min_distant = distant
                    assigned_player = player_id
                    
        return assigned_player
            