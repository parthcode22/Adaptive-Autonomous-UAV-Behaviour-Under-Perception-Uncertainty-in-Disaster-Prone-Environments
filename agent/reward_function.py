"""
reward_function.py — Multi-Component PPO Reward
===================================================
SURE-UAV RL Navigation Agent

Computes the per-step reward for the PPO agent, combining:
    R_safety      — penalizes reckless ACTIONS given current perception
                     state (not the state itself — DANGER/BLIND are
                     expected SAR operating conditions, not failures)
    R_progress    — rewards closing distance to a known signal, or
                     exploration when no signal exists yet
    R_smoothness  — small penalty for switching actions (discourages jitter)
    R_time        — small constant per-step cost (encourages efficiency)
    R_terminal    — one-time crash penalty / victim-found bonus / timeout penalty

R = R_safety + R_progress + R_smoothness + R_time + R_terminal
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from fusion_1.ups_vector import NavigationState
from .action_space import UAVAction

class RewardConstants:
    """All tunable reward weights and magnitudes, centralized."""

    # --- R_safety: action-conditioned on navigation_state ---
    BLIND_CORRECT_ACTION_BONUS    : float =  0.10
    BLIND_RECKLESS_PENALTY        : float = -0.50
    DANGER_FULL_SPEED_PENALTY     : float = -0.30
    DANGER_BACKTRACK_BONUS        : float =  0.05

    # --- R_progress: goal-directed (point A -> point B) ---
    # Reward scales the change in distance-to-goal per step. Since this
    # is now the PRIMARY mission driver, it gets a multiplier so it's
    # comparable in magnitude to the terminal rewards.
    GOAL_PROGRESS_SCALE            : float = 1.0
    MAX_MISSION_DISTANCE           : float = 100.0   # meters — PLACEHOLDER,
                                                       # revisit with real mission specs

    # --- R_progress: victim-seeking is now secondary, small magnitude ---
    # Kept as a minor shaping signal so the agent still prefers investigating
    # nearby signals slightly, without overriding the primary goal-reaching drive.
    SIGNAL_PROGRESS_SCALE          : float = 0.1

    # --- R_smoothness ---
    ACTION_SWITCH_PENALTY         : float = -0.05

    # --- R_time ---
    TIME_PENALTY                  : float = -0.01

    # --- R_terminal ---
    CRASH_PENALTY                 : float = -20.0
    VICTIM_FOUND_BONUS            : float = 10.0       # one-time, mid-episode, does NOT end episode
    GOAL_REACHED_BONUS            : float = 20.0       # episode SUCCESS
    TIMEOUT_PENALTY               : float = -10.0       # episode FAILURE (didn't reach B in time)
    SUCCESS_RADIUS                 : float = 2.0         # meters — within this of point B = goal reached

    # --- Crash detection threshold ---
    CRASH_OBSTACLE_THRESHOLD      : float = 0.85
@dataclass
class RewardContext:
    """
    Everything needed to compute one step's reward.

    Attributes
    ----------
    navigation_state              : current NavigationState from the UPS
    obstacle_confidence            : current obstacle_confidence from the UPS
    action                          : action taken this step
    prev_action                      : action taken the previous step
    distance_to_goal                  : current normalized distance to point B [0,1]
    prev_distance_to_goal               : previous step's normalized distance to point B
    distance_to_last_signal               : current normalized distance to last
                                           victim signal (1.0 = none found yet)
    prev_distance_to_signal                 : previous step's value
    signal_ever_found                          : True if a radar trigger has
                                                occurred at any point this mission
    radar_just_triggered                         : True if radar_flag transitioned
                                                  0->1 THIS step (mid-episode bonus)
    distance_moved                                 : Euclidean distance moved this
                                                    step (meters)
    goal_reached                                     : True if within SUCCESS_RADIUS
                                                      of point B this step
    is_timeout                                         : True if this step ends the
                                                        episode via timeout
    """
    navigation_state          : NavigationState
    obstacle_confidence         : float
    action                       : UAVAction
    prev_action                   : UAVAction
    distance_to_goal                : float
    prev_distance_to_goal             : float
    distance_to_last_signal              : float
    prev_distance_to_signal                : float
    signal_ever_found                        : bool
    radar_just_triggered                       : bool
    distance_moved                               : float
    goal_reached                                   : bool  = False
    is_timeout                                       : bool  = False

def detect_crash(
    obstacle_confidence : float,
    action               : UAVAction,
    consts                : RewardConstants = RewardConstants(),
) -> bool:
    """
    Operational crash definition (no physics sim yet): a crash event is
    flagged when obstacle_confidence is very high AND the agent still
    chose an aggressive forward action — i.e. it ignored a clear warning.

    Once Gazebo/PX4 physics integration exists, this should be replaced
    with real collision detection; this is a placeholder proxy.
    """
    reckless_actions = {UAVAction.FORWARD_NORMAL, UAVAction.FORWARD_SLOW}
    return (
        obstacle_confidence >= consts.CRASH_OBSTACLE_THRESHOLD
        and action in reckless_actions
    )

def _compute_safety_reward(
    nav_state : NavigationState,
    action     : UAVAction,
    consts      : RewardConstants,
) -> float:
    """
    Action-conditioned safety reward. DANGER/BLIND are expected SAR
    operating conditions, NOT penalized by themselves — only reckless
    actions taken while perception is degraded are penalized.
    """
    cautious_actions = {UAVAction.HOVER_OBSERVE, UAVAction.YAW_SCAN_360, UAVAction.BACKTRACK}

    if nav_state == NavigationState.BLIND:
        if action in cautious_actions:
            return consts.BLIND_CORRECT_ACTION_BONUS
        return consts.BLIND_RECKLESS_PENALTY

    if nav_state == NavigationState.DANGER:
        if action == UAVAction.FORWARD_NORMAL:
            return consts.DANGER_FULL_SPEED_PENALTY
        if action == UAVAction.BACKTRACK:
            return consts.DANGER_BACKTRACK_BONUS
        return 0.0   # FORWARD_SLOW, turns, hover, scan — all acceptable cautious progress

    # CAUTION and SAFE — no safety penalty, normal operating conditions
    return 0.0

def _compute_progress_reward(
    ctx    : RewardContext,
    consts  : RewardConstants,
) -> float:
    """
    Primary driver: progress toward point B (goal), scaled by GOAL_PROGRESS_SCALE.
    Secondary, smaller signal: progress toward a known victim signal, scaled
    down so it nudges behavior without overriding the goal-reaching objective.

    These two signals can conflict (e.g. detouring around smoke to stay safe
    temporarily increases distance_to_goal) — this tension is intentional;
    R_safety governs when detours are warranted, R_progress just measures
    net movement efficiency.
    """
    goal_progress = (ctx.prev_distance_to_goal - ctx.distance_to_goal) * consts.GOAL_PROGRESS_SCALE

    signal_progress = 0.0
    if ctx.signal_ever_found:
        signal_progress = (
            (ctx.prev_distance_to_signal - ctx.distance_to_last_signal)
            * consts.SIGNAL_PROGRESS_SCALE
        )

    return goal_progress + signal_progress

def _compute_smoothness_reward(ctx: RewardContext, consts: RewardConstants) -> float:
    """Penalizes switching actions between consecutive steps."""
    if ctx.action != ctx.prev_action:
        return consts.ACTION_SWITCH_PENALTY
    return 0.0


def _compute_time_reward(consts: RewardConstants) -> float:
    """Constant small per-step cost — encourages efficient missions."""
    return consts.TIME_PENALTY


def _compute_terminal_reward(
    ctx          : RewardContext,
    crashed       : bool,
    consts         : RewardConstants,
) -> float:
    """
    One-time terminal/mid-episode events.

    - crash            : episode ends, large penalty
    - radar_just_triggered : mid-episode bonus, does NOT end episode —
                             multiple victims may be found en route to B
    - goal_reached       : episode ends, large bonus (SUCCESS)
    - is_timeout            : episode ends, penalty (FAILURE — didn't reach B)
    """
    reward = 0.0
    if crashed:
        reward += consts.CRASH_PENALTY
    if ctx.radar_just_triggered:
        reward += consts.VICTIM_FOUND_BONUS
    if ctx.goal_reached:
        reward += consts.GOAL_REACHED_BONUS
    if ctx.is_timeout:
        reward += consts.TIMEOUT_PENALTY
    return reward
@dataclass
class RewardResult:
    """Breakdown of the total reward into its components, for logging/debugging."""
    total          : float
    safety          : float
    progress         : float
    smoothness        : float
    time_cost          : float
    terminal            : float
    crashed               : bool
    episode_done           : bool


def compute_reward(
    ctx     : RewardContext,
    consts   : RewardConstants = RewardConstants(),
) -> RewardResult:
    crashed = detect_crash(ctx.obstacle_confidence, ctx.action, consts)

    safety      = _compute_safety_reward(ctx.navigation_state, ctx.action, consts)
    progress     = _compute_progress_reward(ctx, consts)
    smoothness    = _compute_smoothness_reward(ctx, consts)
    time_cost      = _compute_time_reward(consts)
    terminal        = _compute_terminal_reward(ctx, crashed, consts)

    total = safety + progress + smoothness + time_cost + terminal

    # Episode ends on: crash, goal reached (success), or timeout.
    # Victim detection (radar_just_triggered) does NOT end the episode.
    episode_done = crashed or ctx.goal_reached or ctx.is_timeout

    return RewardResult(
        total=total,
        safety=safety,
        progress=progress,
        smoothness=smoothness,
        time_cost=time_cost,
        terminal=terminal,
        crashed=crashed,
        episode_done=episode_done,
    )