"""
TTS语音合成模块
支持多种TTS引擎：本地pyttsx3、在线API、Edge TTS
"""

import os
import hashlib
import threading
import queue
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """TTS引擎类型"""
    PYTTSX3 = "pyttsx3"         # 本地离线引擎
    EDGE_TTS = "edge_tts"       # Edge在线引擎（免费）
    AZURE = "azure"             # Azure认知服务
    BAIDU = "baidu"             # 百度语音


class VoiceGender(Enum):
    """语音性别"""
    MALE = "male"
    FEMALE = "female"


class VoicePriority(Enum):
    """播报优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class TTSRequest:
    """TTS请求"""
    text: str
    priority: VoicePriority = VoicePriority.NORMAL
    workstation_id: Optional[int] = None
    voice_id: Optional[str] = None
    speed: float = 1.0  # 语速 0.5-2.0
    volume: float = 1.0  # 音量 0.0-1.0
    
    # 回调
    on_complete: Optional[Callable] = None
    on_error: Optional[Callable] = None
    
    # 元数据
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TTSConfig:
    """TTS配置"""
    engine: TTSEngine = TTSEngine.PYTTSX3
    
    # 语音设置
    voice_id: str = ""
    voice_gender: VoiceGender = VoiceGender.FEMALE
    default_speed: float = 1.0
    default_volume: float = 1.0
    
    # 缓存设置
    cache_enabled: bool = True
    cache_dir: str = "tts_cache"
    max_cache_size_mb: int = 100
    
    # Azure设置（可选）
    azure_key: str = ""
    azure_region: str = "eastasia"
    
    # 百度设置（可选）
    baidu_app_id: str = ""
    baidu_api_key: str = ""
    baidu_secret_key: str = ""


class AudioCache:
    """音频缓存管理"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        
        # 缓存索引
        self.cache_index: Dict[str, str] = {}
        self._load_index()
    
    def _get_cache_key(self, text: str, voice_id: str = "", speed: float = 1.0) -> str:
        """生成缓存键"""
        content = f"{text}|{voice_id}|{speed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_index(self):
        """加载缓存索引"""
        index_file = self.cache_dir / "index.txt"
        if index_file.exists():
            for line in index_file.read_text().strip().split("\n"):
                if "|" in line:
                    key, path = line.split("|", 1)
                    if (self.cache_dir / path).exists():
                        self.cache_index[key] = path
    
    def _save_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "index.txt"
        lines = [f"{k}|{v}" for k, v in self.cache_index.items()]
        index_file.write_text("\n".join(lines))
    
    def get(self, text: str, voice_id: str = "", speed: float = 1.0) -> Optional[str]:
        """获取缓存的音频文件路径"""
        key = self._get_cache_key(text, voice_id, speed)
        if key in self.cache_index:
            path = self.cache_dir / self.cache_index[key]
            if path.exists():
                return str(path)
        return None
    
    def put(self, text: str, audio_path: str, voice_id: str = "", speed: float = 1.0) -> str:
        """添加到缓存"""
        key = self._get_cache_key(text, voice_id, speed)
        
        # 复制到缓存目录
        ext = Path(audio_path).suffix
        cache_file = f"{key}{ext}"
        cache_path = self.cache_dir / cache_file
        
        import shutil
        shutil.copy(audio_path, cache_path)
        
        self.cache_index[key] = cache_file
        self._save_index()
        
        # 检查缓存大小
        self._check_size()
        
        return str(cache_path)
    
    def _check_size(self):
        """检查并清理缓存"""
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.iterdir() if f.is_file()
        ) / 1024 / 1024  # MB
        
        if total_size > self.max_size_mb:
            # 删除最旧的文件
            files = sorted(
                self.cache_dir.iterdir(),
                key=lambda f: f.stat().st_mtime
            )
            
            for f in files[:len(files) // 2]:
                if f.name != "index.txt":
                    f.unlink()
                    # 从索引移除
                    self.cache_index = {
                        k: v for k, v in self.cache_index.items()
                        if v != f.name
                    }
            
            self._save_index()
            logger.info(f"🧹 清理TTS缓存，当前大小: {total_size:.1f}MB")


class BaseTTSBackend:
    """TTS后端基类"""
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> bool:
        """合成语音"""
        raise NotImplementedError
    
    def speak_sync(self, text: str, **kwargs) -> bool:
        """同步播放"""
        raise NotImplementedError
    
    def get_voices(self) -> List[Dict]:
        """获取可用语音列表"""
        raise NotImplementedError


class Pyttsx3Backend(BaseTTSBackend):
    """pyttsx3本地TTS后端"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.engine = None
        self._init_engine()
    
    def _init_engine(self):
        """初始化引擎"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            
            # 设置语音
            voices = self.engine.getProperty('voices')
            if voices:
                # 选择中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    # 使用默认语音
                    if self.config.voice_gender == VoiceGender.FEMALE:
                        for voice in voices:
                            if 'female' in voice.name.lower():
                                self.engine.setProperty('voice', voice.id)
                                break
            
            # 设置语速和音量
            self.engine.setProperty('rate', 150 * self.config.default_speed)
            self.engine.setProperty('volume', self.config.default_volume)
            
            logger.info("✅ pyttsx3引擎初始化成功")
        except Exception as e:
            logger.error(f"❌ pyttsx3引擎初始化失败: {e}")
            self.engine = None
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> bool:
        """合成语音到文件"""
        if not self.engine:
            return False
        
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            return os.path.exists(output_path)
        except Exception as e:
            logger.error(f"合成失败: {e}")
            return False
    
    def speak_sync(self, text: str, **kwargs) -> bool:
        """同步播放"""
        if not self.engine:
            return False
        
        try:
            speed = kwargs.get('speed', self.config.default_speed)
            volume = kwargs.get('volume', self.config.default_volume)
            
            self.engine.setProperty('rate', 150 * speed)
            self.engine.setProperty('volume', volume)
            
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"播放失败: {e}")
            return False
    
    def get_voices(self) -> List[Dict]:
        """获取可用语音列表"""
        if not self.engine:
            return []
        
        voices = self.engine.getProperty('voices')
        return [
            {
                "id": v.id,
                "name": v.name,
                "languages": v.languages,
                "gender": "female" if "female" in v.name.lower() else "male"
            }
            for v in voices
        ]


class EdgeTTSBackend(BaseTTSBackend):
    """Edge TTS后端（免费在线服务）"""
    
    # 中文语音列表
    CHINESE_VOICES = {
        "zh-CN-XiaoxiaoNeural": {"name": "晓晓", "gender": "female"},
        "zh-CN-XiaoyiNeural": {"name": "晓伊", "gender": "female"},
        "zh-CN-YunjianNeural": {"name": "云健", "gender": "male"},
        "zh-CN-YunxiNeural": {"name": "云希", "gender": "male"},
        "zh-CN-YunyangNeural": {"name": "云扬", "gender": "male"},
        "zh-CN-liaoning-XiaobeiNeural": {"name": "晓北(东北)", "gender": "female"},
        "zh-CN-shaanxi-XiaoniNeural": {"name": "晓妮(陕西)", "gender": "female"},
    }
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.default_voice = "zh-CN-XiaoxiaoNeural"
        
        if config.voice_gender == VoiceGender.MALE:
            self.default_voice = "zh-CN-YunjianNeural"
    
    def synthesize(self, text: str, output_path: str, **kwargs) -> bool:
        """合成语音到文件"""
        try:
            import edge_tts
            
            voice = kwargs.get('voice_id', self.default_voice)
            speed = kwargs.get('speed', self.config.default_speed)
            
            # 转换语速 (edge_tts使用百分比)
            rate = f"+{int((speed - 1) * 100)}%" if speed >= 1 else f"{int((speed - 1) * 100)}%"
            
            async def _synthesize():
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                await communicate.save(output_path)
            
            asyncio.run(_synthesize())
            return os.path.exists(output_path)
            
        except Exception as e:
            logger.error(f"Edge TTS合成失败: {e}")
            return False
    
    def speak_sync(self, text: str, **kwargs) -> bool:
        """同步播放"""
        # 先合成到临时文件，然后播放
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        try:
            if self.synthesize(text, temp_path, **kwargs):
                return self._play_audio(temp_path)
            return False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _play_audio(self, audio_path: str) -> bool:
        """播放音频文件"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
        except:
            # 回退到系统命令
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f"afplay {audio_path}")
            elif system == "Linux":
                os.system(f"mpg123 {audio_path} 2>/dev/null || aplay {audio_path}")
            elif system == "Windows":
                os.system(f"start {audio_path}")
            
            return True
    
    def get_voices(self) -> List[Dict]:
        """获取可用语音列表"""
        return [
            {"id": vid, "name": info["name"], "gender": info["gender"]}
            for vid, info in self.CHINESE_VOICES.items()
        ]


class TTSService:
    """TTS服务"""
    
    def __init__(self, config: TTSConfig = None):
        """
        Args:
            config: TTS配置
        """
        self.config = config or TTSConfig()
        
        # 初始化后端
        self.backend = self._create_backend()
        
        # 初始化缓存
        self.cache = AudioCache(
            self.config.cache_dir,
            self.config.max_cache_size_mb
        ) if self.config.cache_enabled else None
        
        # 播放队列
        self.play_queue = queue.PriorityQueue()
        self.is_playing = False
        self.play_thread: Optional[threading.Thread] = None
        
        # 工位队列（每个工位独立队列）
        self.workstation_queues: Dict[int, queue.PriorityQueue] = {}
        
        # 统计
        self.total_requests = 0
        self.total_played = 0
        self.cache_hits = 0
    
    def _create_backend(self) -> BaseTTSBackend:
        """创建TTS后端"""
        if self.config.engine == TTSEngine.PYTTSX3:
            return Pyttsx3Backend(self.config)
        elif self.config.engine == TTSEngine.EDGE_TTS:
            return EdgeTTSBackend(self.config)
        else:
            logger.warning(f"不支持的引擎: {self.config.engine}, 使用pyttsx3")
            return Pyttsx3Backend(self.config)
    
    def speak(
        self,
        text: str,
        priority: VoicePriority = VoicePriority.NORMAL,
        workstation_id: Optional[int] = None,
        **kwargs
    ):
        """
        异步播放语音（放入队列）
        
        Args:
            text: 文本内容
            priority: 优先级
            workstation_id: 工位ID（可选，用于工位专属播放）
            **kwargs: 其他参数（speed, volume等）
        """
        request = TTSRequest(
            text=text,
            priority=priority,
            workstation_id=workstation_id,
            speed=kwargs.get('speed', self.config.default_speed),
            volume=kwargs.get('volume', self.config.default_volume)
        )
        
        self.total_requests += 1
        
        if workstation_id:
            # 放入工位队列
            if workstation_id not in self.workstation_queues:
                self.workstation_queues[workstation_id] = queue.PriorityQueue()
            self.workstation_queues[workstation_id].put(
                (-priority.value, time.time(), request)
            )
        else:
            # 放入全局队列
            self.play_queue.put((-priority.value, time.time(), request))
        
        # 启动播放线程
        if not self.is_playing:
            self._start_play_thread()
    
    def speak_now(self, text: str, **kwargs) -> bool:
        """
        立即同步播放（阻塞）
        
        Args:
            text: 文本内容
            **kwargs: 其他参数
            
        Returns:
            是否成功
        """
        # 检查缓存
        if self.cache:
            cached_path = self.cache.get(text, kwargs.get('voice_id', ''), kwargs.get('speed', 1.0))
            if cached_path:
                self.cache_hits += 1
                return self._play_file(cached_path)
        
        # 直接播放
        return self.backend.speak_sync(text, **kwargs)
    
    def speak_urgent(self, text: str, workstation_id: Optional[int] = None, **kwargs):
        """紧急播报（最高优先级，打断当前播放）"""
        # TODO: 实现打断功能
        self.speak(text, priority=VoicePriority.URGENT, workstation_id=workstation_id, **kwargs)
    
    def _start_play_thread(self):
        """启动播放线程"""
        if self.play_thread and self.play_thread.is_alive():
            return
        
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
    
    def _play_loop(self):
        """播放循环"""
        while self.is_playing:
            try:
                # 从全局队列获取
                _, _, request = self.play_queue.get(timeout=0.5)
                self._process_request(request)
            except queue.Empty:
                # 检查工位队列
                for ws_id, ws_queue in list(self.workstation_queues.items()):
                    try:
                        _, _, request = ws_queue.get_nowait()
                        self._process_request(request)
                    except queue.Empty:
                        continue
    
    def _process_request(self, request: TTSRequest):
        """处理播放请求"""
        try:
            # 检查缓存
            if self.cache:
                cached_path = self.cache.get(request.text, request.voice_id or '', request.speed)
                if cached_path:
                    self.cache_hits += 1
                    self._play_file(cached_path)
                    self.total_played += 1
                    if request.on_complete:
                        request.on_complete()
                    return
            
            # 合成并播放
            if self.backend.speak_sync(request.text, speed=request.speed, volume=request.volume):
                self.total_played += 1
                if request.on_complete:
                    request.on_complete()
            else:
                if request.on_error:
                    request.on_error("播放失败")
                    
        except Exception as e:
            logger.error(f"处理TTS请求失败: {e}")
            if request.on_error:
                request.on_error(str(e))
    
    def _play_file(self, audio_path: str) -> bool:
        """播放音频文件"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
        except:
            # 回退
            import platform
            system = platform.system()
            
            if system == "Darwin":
                os.system(f"afplay {audio_path}")
            elif system == "Linux":
                os.system(f"mpg123 {audio_path} 2>/dev/null")
            elif system == "Windows":
                os.system(f"start {audio_path}")
            
            return True
    
    def synthesize_to_file(self, text: str, output_path: str, **kwargs) -> bool:
        """
        合成语音到文件
        
        Args:
            text: 文本内容
            output_path: 输出文件路径
            **kwargs: 其他参数
            
        Returns:
            是否成功
        """
        return self.backend.synthesize(text, output_path, **kwargs)
    
    def get_voices(self) -> List[Dict]:
        """获取可用语音列表"""
        return self.backend.get_voices()
    
    def stop(self):
        """停止服务"""
        self.is_playing = False
        if self.play_thread:
            self.play_thread.join(timeout=2)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "engine": self.config.engine.value,
            "total_requests": self.total_requests,
            "total_played": self.total_played,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_played, 1) * 100,
            "pending_global": self.play_queue.qsize(),
            "workstation_queues": len(self.workstation_queues)
        }


# 预定义语音消息
class VoiceMessages:
    """预定义语音消息"""
    
    # 系统消息
    SYSTEM_READY = "系统已就绪，请开始训练"
    SYSTEM_ERROR = "系统出现异常，请联系管理员"
    
    # 训练消息
    TRAINING_START = "训练开始，请保持标准姿势"
    TRAINING_PAUSE = "训练暂停"
    TRAINING_RESUME = "训练继续"
    TRAINING_END = "本次训练结束"
    
    # 分数消息
    SCORE_EXCELLENT = "非常好！动作标准，得分优秀"
    SCORE_GOOD = "做得不错，继续保持"
    SCORE_FAIR = "动作需要改进"
    SCORE_POOR = "动作不合格，请注意纠正"
    
    # 动作提示
    POSTURE_CORRECT = "姿势正确"
    POSTURE_ADJUST = "请调整姿势"
    TRIGGER_DISCIPLINE = "注意扳机纪律，手指离开扳机"
    AIM_ADJUST = "调整瞄准姿势"
    STABILITY_IMPROVE = "保持稳定，放松呼吸"
    
    # 警告消息
    WARNING_FINGER_ON_TRIGGER = "警告！手指不要放在扳机上"
    WARNING_MUZZLE_DIRECTION = "警告！注意枪口方向"
    WARNING_STANCE_UNSTABLE = "站姿不稳，请调整"
    
    # 鼓励消息
    ENCOURAGE_KEEP_GOING = "加油！继续努力"
    ENCOURAGE_ALMOST_THERE = "马上就达标了，再坚持一下"
    ENCOURAGE_WELL_DONE = "太棒了！你做到了"
    
    # 阶段消息
    STAGE_READY_FOR_LIVE = "恭喜！你已达到实弹训练标准"
    STAGE_NEED_MORE_PRACTICE = "还需要继续练习"
    
    @classmethod
    def get_score_message(cls, score: float) -> str:
        """根据分数获取消息"""
        if score >= 90:
            return cls.SCORE_EXCELLENT
        elif score >= 80:
            return cls.SCORE_GOOD
        elif score >= 70:
            return cls.SCORE_FAIR
        else:
            return cls.SCORE_POOR


# 创建默认TTS服务实例
default_tts = TTSService(TTSConfig(engine=TTSEngine.PYTTSX3))


# 便捷函数
def speak(text: str, **kwargs):
    """播放语音（使用默认服务）"""
    default_tts.speak(text, **kwargs)


def speak_now(text: str, **kwargs) -> bool:
    """立即播放语音"""
    return default_tts.speak_now(text, **kwargs)


def speak_warning(text: str, workstation_id: Optional[int] = None):
    """播放警告"""
    default_tts.speak_urgent(text, workstation_id=workstation_id)


# 使用示例
if __name__ == "__main__":
    # 创建TTS服务
    config = TTSConfig(
        engine=TTSEngine.PYTTSX3,  # 或 TTSEngine.EDGE_TTS
        voice_gender=VoiceGender.FEMALE,
        default_speed=1.0
    )
    
    tts = TTSService(config)
    
    # 获取可用语音
    voices = tts.get_voices()
    print(f"可用语音: {len(voices)}")
    for v in voices[:5]:
        print(f"  - {v}")
    
    # 播放测试
    print("\n🔊 播放测试...")
    
    # 同步播放
    tts.speak_now("系统初始化完成，欢迎使用智能枪械训练系统")
    
    # 异步播放（队列）
    tts.speak(VoiceMessages.TRAINING_START)
    tts.speak(VoiceMessages.POSTURE_CORRECT)
    tts.speak(VoiceMessages.SCORE_GOOD)
    
    # 等待播放完成
    time.sleep(10)
    
    # 统计
    print(f"\n📊 统计: {tts.get_stats()}")
    
    tts.stop()

