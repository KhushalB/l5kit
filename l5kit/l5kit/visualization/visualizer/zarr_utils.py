from typing import List, Tuple, no_type_check, NamedTuple

import numpy as np

from l5kit.data import ChunkedDataset
from l5kit.data.filter import (filter_agents_by_frames, filter_agents_by_labels, filter_tl_faces_by_frames,
                               filter_tl_faces_by_status)
from l5kit.data.labels import PERCEPTION_LABELS
from l5kit.data.map_api import MapAPI, TLFacesColors
from l5kit.geometry import transform_points
from l5kit.rasterization.box_rasterizer import get_box_world_coords, get_ego_as_agent
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.sampling import get_relative_poses
from l5kit.simulation.unroll import SimulationOutput, UnrollInputOutput
from l5kit.visualization.visualizer.common import (AgentVisualization, EgoVisualization, FrameVisualization,
                                                   MapElementVisualization, TrajectoryVisualization)


# TODO: this should not be here (maybe a config?)
COLORS = {
    TLFacesColors.GREEN.name: "#95e494",
    TLFacesColors.RED.name: "#f9a589",
    TLFacesColors.YELLOW.name: "#FFFF66",
    "LANE_DEFAULT": "#BFBFBF",
    "AGENT_DEFAULT": "#1F77B4",
    "PERCEPTION_LABEL_CAR": "#1F77B4",
    "PERCEPTION_LABEL_CYCLIST": "#CC33FF",
    "PERCEPTION_LABEL_PEDESTRIAN": "#66CCFF",
}


def _get_frame_trajectories(frames: np.ndarray, agents_frames: List[np.ndarray], track_ids: np.ndarray,
                            frame_index: int, ego_length: int, agent_length: int) -> List[TrajectoryVisualization]:
    """Get trajectories (ego and agents) starting at frame_index.
    Ego's trajectory will be named ego_trajectory while agents' agent_trajectory

    :param frames: all frames from the scene
    :param agents_frames: all agents from the scene as a list of array (one per frame)
    :param track_ids: allowed tracks ids we want to build trajectory for
    :param frame_index: index of the frame (trajectory will start from this frame)
    :param ego_length: length of ego trajectory to extract
    :param agent_length: length of agent trajectory to extract
    :return: a list of trajectory for visualisation
    """

    traj_visualisation: List[TrajectoryVisualization] = []
    if agent_length > 0:
        for track_id in track_ids:
            # Note (@lberg): this is not making it relative (note eye and 0 yaw)
            pos, *_, avail = get_relative_poses(agent_length, frames[frame_index: frame_index + agent_length],
                                                track_id, agents_frames[frame_index: frame_index + agent_length],
                                                np.eye(3), 0)
            traj_visualisation.append(TrajectoryVisualization(xs=pos[avail > 0, 0],
                                                              ys=pos[avail > 0, 1],
                                                              color=COLORS["AGENT_DEFAULT"],
                                                              legend_label="agent_trajectory",
                                                              track_id=int(track_id)))

    if ego_length > 0:
        pos, *_, avail = get_relative_poses(ego_length, frames[frame_index: frame_index + ego_length],
                                            None, agents_frames[frame_index: frame_index + ego_length],
                                            np.eye(3), 0)
        traj_visualisation.append(TrajectoryVisualization(xs=pos[avail > 0, 0],
                                                          ys=pos[avail > 0, 1],
                                                          color="red",
                                                          legend_label="ego_trajectory",
                                                          track_id=-1))

    return traj_visualisation


@no_type_check
def is_lane_painted(mapAPI: MapAPI, element_id: str) -> bool:
    """Return True if the lane with the element_id is a real lane (i.e. it's painted on the road).
    We check both the divider type and the role in the junction for this

    :param mapAPI: mapAPI object
    :param element_id: the lane element id
    :return:
    """
    element = mapAPI[element_id]
    if not mapAPI.is_lane(element):
        raise ValueError(f"{element_id} is not a lane")

    lane = element.element.lane
    left_bound = lane.left_boundary
    right_bound = lane.right_boundary

    # check if it's a real line
    if 1 in left_bound.divider_type or 1 in right_bound.divider_type:  # from L460 road_network.proto
        return False
    # remove additional curve sections
    if lane.turn_type_in_parent_junction in [2, 3, 4, 5, 6]:  # L427 of road_network.proto
        return False

    return True


def _get_frame_data(mapAPI: MapAPI, frame: np.ndarray, agents_frame: np.ndarray,
                    tls_frame: np.ndarray) -> FrameVisualization:
    """Get visualisation objects for the current frame.

    :param mapAPI: mapAPI object (used for lanes, crosswalks etc..)
    :param frame: the current frame (used for ego)
    :param agents_frame: agents in this frame
    :param tls_frame: the tls of this frame
    :return: A FrameVisualization object. NOTE: trajectory are not included here
    """
    ego_xy = frame["ego_translation"][:2]

    #################
    # plot map patches
    map_patches_vis: List[MapElementVisualization] = []
    # this will have priority in visualisation
    map_patches_vis_lane_prio: List[MapElementVisualization] = []
    map_lines_vis: List[MapElementVisualization] = []

    lane_indices = indices_in_bounds(ego_xy, mapAPI.bounds_info["lanes"]["bounds"], 50)
    active_tl_ids = set(filter_tl_faces_by_status(tls_frame, "ACTIVE")["face_id"].tolist())

    for idx, lane_idx in enumerate(lane_indices):
        lane_idx = mapAPI.bounds_info["lanes"]["ids"][lane_idx]

        lane_tl_ids = set(mapAPI.get_lane_traffic_control_ids(lane_idx))
        lane_colour = COLORS["LANE_DEFAULT"]
        for tl_id in lane_tl_ids.intersection(active_tl_ids):
            lane_colour = COLORS[mapAPI.get_color_for_face(tl_id)]

        lane_coords = mapAPI.get_lane_coords(lane_idx)
        left_lane = lane_coords["xyz_left"][:, :2]
        right_lane = lane_coords["xyz_right"][::-1, :2]

        if lane_colour == COLORS["LANE_DEFAULT"]:
            map_patches_vis.append(MapElementVisualization(xs=np.hstack((left_lane[:, 0], right_lane[:, 0])),
                                                           ys=np.hstack((left_lane[:, 1], right_lane[:, 1])),
                                                           color=lane_colour, alpha=1.0))
        else:
            map_patches_vis_lane_prio.append(MapElementVisualization(xs=np.hstack((left_lane[:, 0], right_lane[:, 0])),
                                                                     ys=np.hstack((left_lane[:, 1], right_lane[:, 1])),
                                                                     color=lane_colour, alpha=1.0))

        # add bounds for painted lanes
        if is_lane_painted(mapAPI, lane_idx):
            map_lines_vis.append(MapElementVisualization(xs=left_lane[:, 0],
                                                         ys=left_lane[:, 1],
                                                         color="white", alpha=1.0))
            map_lines_vis.append(MapElementVisualization(xs=right_lane[:, 0],
                                                         ys=right_lane[:, 1],
                                                         color="white", alpha=1.0))

    # add lanes with TLS
    map_patches_vis.extend(map_patches_vis_lane_prio)

    #################
    # plot crosswalks
    crosswalk_indices = indices_in_bounds(ego_xy, mapAPI.bounds_info["crosswalks"]["bounds"], 50)
    for idx in crosswalk_indices:
        crosswalk = mapAPI.get_crosswalk_coords(mapAPI.bounds_info["crosswalks"]["ids"][idx])
        map_patches_vis.append(MapElementVisualization(xs=crosswalk["xyz"][:, 0],
                                                       ys=crosswalk["xyz"][:, 1],
                                                       color="#c0c170", alpha=1.0))

    #################
    # plot ego and agents
    agents_frame = np.insert(agents_frame, 0, get_ego_as_agent(frame))
    box_world_coords = get_box_world_coords(agents_frame)

    # ego
    ego_vis = EgoVisualization(xs=box_world_coords[0, :, 0], ys=box_world_coords[0, :, 1],
                               color="#B53331", alpha=1.0, center_x=agents_frame["centroid"][0, 0],
                               center_y=agents_frame["centroid"][0, 1])

    # agents
    agents_frame = agents_frame[1:]
    box_world_coords = box_world_coords[1:]

    agents_vis: List[AgentVisualization] = []
    for agent, box_coord in zip(agents_frame, box_world_coords):
        label_index = np.argmax(agent["label_probabilities"])
        agents_vis.append(AgentVisualization(xs=box_coord[..., 0],
                                             ys=box_coord[..., 1],
                                             color=COLORS["AGENT_DEFAULT"],
                                             alpha=1.0,
                                             track_id=agent["track_id"],
                                             agent_type=PERCEPTION_LABELS[label_index],
                                             prob=agent["label_probabilities"][label_index]))

    return FrameVisualization(ego=[ego_vis], agents=agents_vis, map_patches=map_patches_vis,
                              map_lines=map_lines_vis, trajectories=[])


class VisualizerZarrConfig(NamedTuple):
    """Visualizer configuration when converting a Zarr

    :param ego_trajectory_length: length of the trajectory for ego, 0 to disable
    :param agents_trajectory_length: length of the trajectory for agents, 0 to disable
    :param agents_threshold: threshold over agents
    """
    ego_trajectory_length: int
    agents_trajectory_length: int
    agents_threshold: float


def zarr_to_visualizer_scene(scene_dataset: ChunkedDataset, mapAPI: MapAPI,
                             visualizer_config: VisualizerZarrConfig) -> List[FrameVisualization]:
    """Convert a zarr scene into a list of FrameVisualization which can be used by the visualiser.
    The visualised trajectory are extracted from future frames.

    :param scene_dataset: a scene dataset. This must contain a single scene
    :param mapAPI: mapAPI object
    :param visualizer_config: config for the visualizer
    :return: a list of FrameVisualization objects
    """
    if len(scene_dataset.scenes) != 1:
        raise ValueError(f"we can convert only a single scene, found {len(scene_dataset.scenes)}")

    frames = scene_dataset.frames
    agents_frames = filter_agents_by_frames(frames, scene_dataset.agents)
    tls_frames = filter_tl_faces_by_frames(frames, scene_dataset.tl_faces)

    frames_vis: List[FrameVisualization] = []
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        tls_frame = tls_frames[frame_idx]

        agents_frame = agents_frames[frame_idx]
        agents_frame = filter_agents_by_labels(agents_frame, visualizer_config.agents_threshold)

        frame_vis = _get_frame_data(mapAPI, frame, agents_frame, tls_frame)

        traj_vis = _get_frame_trajectories(frames, agents_frames, agents_frame["track_id"], frame_idx,
                                           ego_length=visualizer_config.ego_trajectory_length,
                                           agent_length=visualizer_config.agents_trajectory_length)
        frame_vis = FrameVisualization(ego=frame_vis.ego, agents=frame_vis.agents,
                                       map_patches=frame_vis.map_patches,
                                       map_lines=frame_vis.map_lines,
                                       trajectories=traj_vis)
        frames_vis.append(frame_vis)

    return frames_vis


def _get_in_out_as_trajectories(in_out: UnrollInputOutput) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the input (log-replayed) and output (simulated) trajectories into world space.
    Apply availability on the log-replayed one

    :param in_out: an UnrollInputOutput object
    :return: the replayed and simulated trajectory as numpy arrays
    """
    replay_traj = transform_points(in_out.inputs["target_positions"],
                                   in_out.inputs["world_from_agent"])
    replay_traj = replay_traj[in_out.inputs["target_availabilities"] > 0]
    sim_traj = transform_points(in_out.outputs["positions"],
                                in_out.inputs["world_from_agent"])

    return replay_traj, sim_traj


def simulation_out_to_visualizer_scene(sim_out: SimulationOutput, mapAPI: MapAPI) -> List[FrameVisualization]:
    """Convert a simulation output into a scene we can visualize.
    The scene will be created out of the simulated dataset, and will feature trajectories if available.

    :param sim_out: the simulation output
    :param mapAPI: a MapAPI object
    :return: a list of FrameVisualization for the scene
    """
    # sim info
    frames_sim = sim_out.simulated_ego
    agents_frames_sim = filter_agents_by_frames(frames_sim, sim_out.simulated_agents)
    tls_frames_sim = filter_tl_faces_by_frames(frames_sim, sim_out.simulated_dataset.dataset.tl_faces)
    agents_th = sim_out.simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]

    agents_ins_outs = sim_out.agents_ins_outs
    ego_ins_outs = sim_out.ego_ins_outs

    frames_vis: List[FrameVisualization] = []
    for frame_idx in range(len(frames_sim)):
        trajectories = []

        # simulation
        frame_sim = frames_sim[frame_idx]
        tls_frame_sim = tls_frames_sim[frame_idx]
        agents_frame_sim = agents_frames_sim[frame_idx]
        agents_frame_sim = filter_agents_by_labels(agents_frame_sim, agents_th)

        frame_vis_sim = _get_frame_data(mapAPI, frame_sim, agents_frame_sim, tls_frame_sim)

        if not sim_out.sim_cfg.use_ego_gt:
            _, sim_traj = _get_in_out_as_trajectories(ego_ins_outs[frame_idx])
            trajectories.append(TrajectoryVisualization(xs=sim_traj[:, 0], ys=sim_traj[:, 1],
                                                        color="#B53331",
                                                        legend_label="ego_simulated",
                                                        track_id=-1))
        if not sim_out.sim_cfg.use_agents_gt:
            agents_in_out = agents_ins_outs[frame_idx]
            for agent_in_out in agents_in_out:
                track_id = agent_in_out.inputs["track_id"]
                _, sim_traj = _get_in_out_as_trajectories(agent_in_out)
                trajectories.append(TrajectoryVisualization(xs=sim_traj[:, 0], ys=sim_traj[:, 1],
                                                            color=COLORS["AGENT_DEFAULT"],
                                                            legend_label="agent_simulated",
                                                            track_id=track_id))

        frame_vis = FrameVisualization(ego=frame_vis_sim.ego, agents=frame_vis_sim.agents,
                                       map_patches=frame_vis_sim.map_patches,
                                       map_lines=frame_vis_sim.map_lines,
                                       trajectories=trajectories)

        frames_vis.append(frame_vis)

    return frames_vis
