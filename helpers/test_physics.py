import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])) # run from anywhere: put the repo root on the path

import numpy as np
import cv2
from helpers.ogbench_methods import OGBenchMethods

ENV_NAME = 'cube-single-play-singletask-v0'
CUBE_HINTS = ('cube', 'block', 'object', 'obj')
SPEED = 0.5
SCALE = 3
FPS = 30
WIN = 'teleop  |  WASD=xy  QE=z  ZC=yaw  SPACE=grip  R=reset  ESC=quit'

def find_cube(env):
    try:
        import mujoco
        m = env.unwrapped.model
        for i in range(m.nbody):
            n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '').lower()
            if any(h in n for h in CUBE_HINTS):
                return i
    except Exception:
        pass
    return None

def main():
    env, _, _ = OGBenchMethods.load_ogbench(ENV_NAME)

    action_dim = env.action_space.shape[0]
    cube_id = find_cube(env)

    IX, IY, IZ = 0, 1, 2
    IYAW = 3 if action_dim >= 5 else None
    IGRIP = action_dim - 1

    obs, info = env.reset()
    frame = env.render()
    if frame is None:
        env.close()
        return

    def cube_pos():
        if cube_id is None:
            return None
        return np.array(env.unwrapped.data.xpos[cube_id]).copy()

    cube_ref = cube_pos()
    move = np.zeros(action_dim, dtype=np.float32)
    grip = -1.0

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    h, w = np.asarray(frame).shape[:2]
    cv2.resizeWindow(WIN, w * SCALE, h * SCALE)

    while True:
        action = move.copy()
        action[IGRIP] = grip

        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()

        if frame is not None:
            f = np.asarray(frame, dtype=np.uint8)
            disp = f[..., ::-1]

            moved = 0.0
            cp = cube_pos()
            if cp is not None and cube_ref is not None:
                moved = float(np.linalg.norm(cp - cube_ref))

            txt = (f'grip={"CLOSED" if grip > 0 else "open"}  '
                   f'cube_moved={moved:.4f}  r={reward:+.2f}')
            disp = cv2.resize(disp, (w * SCALE, h * SCALE),
                              interpolation=cv2.INTER_NEAREST)
            colour = (0, 220, 0) if moved > 1e-3 else (220, 220, 220)
            cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, colour, 2, cv2.LINE_AA)
            cv2.imshow(WIN, disp)

        k = cv2.waitKey(max(1, int(1000 / FPS))) & 0xFF

        move[:] = 0

        if k == 27:
            break
        elif k == ord('w'):
            move[IX] = +SPEED
        elif k == ord('s'):
            move[IX] = -SPEED
        elif k == ord('d'):
            move[IY] = +SPEED
        elif k == ord('a'):
            move[IY] = -SPEED
        elif k == ord('q'):
            move[IZ] = +SPEED
        elif k == ord('e'):
            move[IZ] = -SPEED
        elif k == ord('z') and IYAW is not None:
            move[IYAW] = -SPEED
        elif k == ord('c') and IYAW is not None:
            move[IYAW] = +SPEED
        elif k == ord(' '):
            grip = -grip
        elif k == ord('r'):
            obs, info = env.reset()
            cube_ref = cube_pos()

    cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    main()