import sys
from pathlib import Path

# Ensure repo root is importable before importing our package modules
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Switch4EmbodiedAI.modules.pipeline import StreamToRobotPipeline, PipelineConfig
from Switch4EmbodiedAI.modules.UDPcomm_module import UDPComm


def main():
    cfg = PipelineConfig()
    pipeline = StreamToRobotPipeline(cfg)
    client = None
    if cfg.udp_enabled:
        client = UDPComm(cfg.udp_ip, cfg.udp_send_port)
    pipeline.start()
    try:
        while True:
            out = pipeline.run_once()
            if out is None:
                continue
            if client is not None:
                if "qpos" in out:
                    arr = out["qpos"].astype("float32").flatten()
                else:
                    md = out["motion_data"]
                    # # pack root_pos (3), root_rot (4), dof_pos (rest)
                    # root_pos = md["root_pos"].reshape(-1)
                    # root_rot = md["root_rot"].reshape(-1)
                    # dof_pos = md["dof_pos"].reshape(-1)
                    # arr = np.concatenate([root_pos, root_rot, dof_pos]).astype("float32")
                client.send_message(md, cfg.udp_ip, cfg.udp_send_port)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()


