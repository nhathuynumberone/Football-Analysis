from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from cam_movement_est import CamMovementEst
from view_trans import ViewTransformer
from speed_dist_est import SpeedAndDistance_Estimator


def main():
    video_frames = read_video('input_videos/football_match1080p.mp4')
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)

    cam_movement_est = CamMovementEst(video_frames[0])
    camera_movement_per_frame = cam_movement_est.get_camera_movement(video_frames,read_from_stub=True,stub_path='stubs/camera_movement_stub.pkl')
    cam_movement_est.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    view_trans = ViewTransformer()
    view_trans.add_transformed_position_to_tracks(tracks)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_dist_est = SpeedAndDistance_Estimator()
    speed_dist_est.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    success_pass = [0]

    last_player_id = None
    last_team = None

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            current_team = tracks['players'][frame_num][assigned_player]['team']
            team_ball_control.append(current_team)
            
            if (
            last_player_id is not None and  # Ensure there was a previous player
            last_player_id != assigned_player and last_team == current_team 
            ):
                success_pass.append(current_team)  # Successful pass
            else:
                success_pass.append(0)  # No pass or invalid pass
            last_player_id = assigned_player
            last_team = current_team    
            
        else:
            team_ball_control.append(team_ball_control[-1])
            success_pass.append(0)

    team_ball_control= np.array(team_ball_control)
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control, success_pass)
    output_video_frames = cam_movement_est.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    speed_dist_est.draw_speed_and_distance(output_video_frames,tracks)
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
