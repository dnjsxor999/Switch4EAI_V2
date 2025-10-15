from dataclasses import dataclass

from Switch4EmbodiedAI.modules.stream_module import SimpleStreamModuleConfig
from Switch4EmbodiedAI.modules.gvhmr_realtime import GVHMRRealtimeConfig
from Switch4EmbodiedAI.modules.gmr_retarget import GMRConfig


@dataclass
class AppConfig:
    stream: SimpleStreamModuleConfig = SimpleStreamModuleConfig()
    gvhmr: GVHMRRealtimeConfig = GVHMRRealtimeConfig()
    gmr: GMRConfig = GMRConfig()


