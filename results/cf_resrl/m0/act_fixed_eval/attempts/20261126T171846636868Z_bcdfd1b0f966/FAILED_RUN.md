# ACT fixed-checkpoint evaluation launch blocked before rollout

The worker exited while checking the pip environment lock. It captured the lock after replacing the launcher's Python import path with the pinned ResFiT and LeRobot checkout paths. The recorded difference consists of the host ROS packages that were visible to the original lock command. Capturing the lock at the launcher before replacing the worker's import path matches the formal ACT training command. The failed attempt executed zero evaluation episodes.
