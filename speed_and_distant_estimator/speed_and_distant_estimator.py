import sys
import cv2
sys.path.append('../')
from utils import measure_distant, get_foot_position
class SpeedAndDistantEstimator:
    def __init__(self) -> None:
        self.frame_window = 5
        self.frame_rate = 24
    
    def add_speed_and_distant_to_tracks(self, tracks):
        total_distant_covered = {}
        for object, object_tracks in tracks.items():
            if object in ('ball', 'referees'):
                continue
            num_frames = len(object_tracks)
            for frame_num in range(0, num_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, num_frames - 1)
                
                for track_id,_ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue
                
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']
                    
                    if start_position is None or end_position is None:
                        continue
                    
                    distant_covered = measure_distant(start_position, end_position)
                    time_ellapsed = (last_frame - frame_num)/self.frame_rate
                    speed = distant_covered/time_ellapsed
                    speed_km_per_hour = speed*3.6
                    
                    if object not in total_distant_covered:
                        total_distant_covered[object] = {}
                        
                    if track_id not in total_distant_covered[object]:
                        total_distant_covered[object][track_id] = 0
                        
                    total_distant_covered[object][track_id] += distant_covered
                    
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distant'] = total_distant_covered[object][track_id]

    def draw_speed_and_distant(self, video_frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            for object, object_track in tracks.items():
                if object in ('ball', 'referees'):
                    continue
                for _, track_info in object_track[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed', None)
                        distant = track_info.get('distant', None)
                        if speed is None or distant is None:
                            continue
                        
                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        
                        position = tuple(map(int, position))
                        cv2.putText(frame, f"Speed: {speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,0), 2)
                        cv2.putText(frame, f"Distant: {speed:.2f} m", (position[0], position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(0,0,0), 2)
                    
            output_frames.append(frame)
        return output_frames