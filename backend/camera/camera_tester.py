"""
海康威视摄像头连接测试工具
用于测试真实摄像头的连接、画质、延迟等指标
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
import logging
import json
import os

from .hikvision import HikvisionCameraConfig, HikvisionCamera, CameraStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionTestResult:
    """连接测试结果"""
    camera_id: int
    ip: str
    success: bool
    connection_time_ms: float = 0.0
    error_message: str = ""
    
    # 视频参数
    resolution: Tuple[int, int] = (0, 0)
    actual_fps: float = 0.0
    codec: str = ""
    
    # 延迟测试
    latency_ms: float = 0.0
    
    # 画质评估
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    noise_level: float = 0.0
    
    # 稳定性测试
    frames_received: int = 0
    frames_dropped: int = 0
    stability_score: float = 0.0
    
    test_timestamp: datetime = field(default_factory=datetime.now)
    test_duration_seconds: float = 0.0


@dataclass
class BatchTestResult:
    """批量测试结果"""
    total_cameras: int
    success_count: int
    failed_count: int
    results: List[ConnectionTestResult]
    test_timestamp: datetime
    total_duration_seconds: float
    
    @property
    def success_rate(self) -> float:
        if self.total_cameras == 0:
            return 0.0
        return self.success_count / self.total_cameras * 100


class CameraQualityAnalyzer:
    """摄像头画质分析器"""
    
    @staticmethod
    def analyze_brightness(frame: np.ndarray) -> float:
        """分析亮度 (0-100)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) / 255 * 100
    
    @staticmethod
    def analyze_contrast(frame: np.ndarray) -> float:
        """分析对比度 (0-100)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray) / 128 * 100
    
    @staticmethod
    def analyze_sharpness(frame: np.ndarray) -> float:
        """分析清晰度 (基于Laplacian方差)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # 归一化到0-100
        return min(variance / 500 * 100, 100)
    
    @staticmethod
    def analyze_noise(frame: np.ndarray) -> float:
        """分析噪声水平 (0-100, 越低越好)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用高通滤波器检测噪声
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blur)
        noise_level = np.mean(noise)
        return min(noise_level / 30 * 100, 100)
    
    @classmethod
    def analyze_frame(cls, frame: np.ndarray) -> Dict[str, float]:
        """完整的帧质量分析"""
        return {
            "brightness": cls.analyze_brightness(frame),
            "contrast": cls.analyze_contrast(frame),
            "sharpness": cls.analyze_sharpness(frame),
            "noise_level": cls.analyze_noise(frame)
        }


class HikvisionCameraTester:
    """海康威视摄像头测试器"""
    
    def __init__(self, test_duration: float = 5.0):
        """
        Args:
            test_duration: 每个摄像头测试时长（秒）
        """
        self.test_duration = test_duration
        self.quality_analyzer = CameraQualityAnalyzer()
    
    def test_connection(self, config: HikvisionCameraConfig) -> ConnectionTestResult:
        """
        测试单个摄像头连接
        
        Args:
            config: 摄像头配置
            
        Returns:
            测试结果
        """
        result = ConnectionTestResult(
            camera_id=config.camera_id,
            ip=config.ip,
            success=False
        )
        
        logger.info(f"🔍 测试摄像头 {config.camera_id} ({config.ip})...")
        
        start_time = time.time()
        camera = None
        
        try:
            # 创建摄像头实例
            camera = HikvisionCamera(
                config=config,
                target_fps=10,
                reconnect_interval=2,
                max_reconnect_attempts=2
            )
            
            # 测试连接
            connect_start = time.time()
            if not camera.connect():
                result.error_message = "连接失败"
                return result
            
            result.connection_time_ms = (time.time() - connect_start) * 1000
            
            # 获取视频参数
            if camera.cap:
                result.resolution = (
                    int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
                
                fourcc = int(camera.cap.get(cv2.CAP_PROP_FOURCC))
                result.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # 启动采集
            camera.start()
            time.sleep(0.5)  # 等待稳定
            
            # 采集测试帧
            frames_received = 0
            quality_samples = []
            latency_samples = []
            
            test_end = time.time() + self.test_duration
            
            while time.time() < test_end:
                frame_start = time.time()
                frame_obj = camera.get_frame(timeout=1.0)
                
                if frame_obj:
                    frames_received += 1
                    latency_samples.append((time.time() - frame_start) * 1000)
                    
                    # 每秒分析一次画质
                    if len(quality_samples) < int(self.test_duration):
                        quality = self.quality_analyzer.analyze_frame(frame_obj.frame)
                        quality_samples.append(quality)
                
                time.sleep(0.05)
            
            # 计算结果
            result.success = True
            result.frames_received = frames_received
            result.actual_fps = frames_received / self.test_duration
            
            if latency_samples:
                result.latency_ms = np.mean(latency_samples)
            
            if quality_samples:
                result.brightness = np.mean([q["brightness"] for q in quality_samples])
                result.contrast = np.mean([q["contrast"] for q in quality_samples])
                result.sharpness = np.mean([q["sharpness"] for q in quality_samples])
                result.noise_level = np.mean([q["noise_level"] for q in quality_samples])
            
            # 计算稳定性分数
            expected_frames = self.test_duration * 10  # 目标10fps
            result.stability_score = min(frames_received / expected_frames * 100, 100)
            
            logger.info(f"✅ 摄像头 {config.camera_id} 测试通过: "
                       f"{result.resolution[0]}x{result.resolution[1]}@{result.actual_fps:.1f}fps, "
                       f"延迟{result.latency_ms:.1f}ms")
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"❌ 摄像头 {config.camera_id} 测试失败: {e}")
        
        finally:
            if camera:
                camera.stop()
            
            result.test_duration_seconds = time.time() - start_time
        
        return result
    
    def test_batch(
        self,
        configs: List[HikvisionCameraConfig],
        parallel: bool = True,
        max_workers: int = 10
    ) -> BatchTestResult:
        """
        批量测试摄像头
        
        Args:
            configs: 摄像头配置列表
            parallel: 是否并行测试
            max_workers: 最大并行数
            
        Returns:
            批量测试结果
        """
        start_time = time.time()
        results = []
        
        logger.info(f"🚀 开始批量测试 {len(configs)} 个摄像头...")
        
        if parallel:
            # 并行测试
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.test_connection, config): config
                    for config in configs
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # 进度显示
                    done = len(results)
                    total = len(configs)
                    success = sum(1 for r in results if r.success)
                    logger.info(f"📊 进度: {done}/{total}, 成功: {success}")
        else:
            # 串行测试
            for i, config in enumerate(configs):
                result = self.test_connection(config)
                results.append(result)
                logger.info(f"📊 进度: {i+1}/{len(configs)}")
        
        # 汇总结果
        success_count = sum(1 for r in results if r.success)
        
        batch_result = BatchTestResult(
            total_cameras=len(configs),
            success_count=success_count,
            failed_count=len(configs) - success_count,
            results=results,
            test_timestamp=datetime.now(),
            total_duration_seconds=time.time() - start_time
        )
        
        logger.info(f"✅ 批量测试完成: {success_count}/{len(configs)} 成功 "
                   f"({batch_result.success_rate:.1f}%), "
                   f"耗时 {batch_result.total_duration_seconds:.1f}秒")
        
        return batch_result
    
    def generate_report(
        self,
        batch_result: BatchTestResult,
        output_file: str = "camera_test_report.json"
    ) -> str:
        """
        生成测试报告
        
        Args:
            batch_result: 批量测试结果
            output_file: 输出文件路径
            
        Returns:
            报告文件路径
        """
        report = {
            "summary": {
                "total_cameras": batch_result.total_cameras,
                "success_count": batch_result.success_count,
                "failed_count": batch_result.failed_count,
                "success_rate": batch_result.success_rate,
                "test_timestamp": batch_result.test_timestamp.isoformat(),
                "total_duration_seconds": batch_result.total_duration_seconds
            },
            "results": []
        }
        
        for r in batch_result.results:
            report["results"].append({
                "camera_id": r.camera_id,
                "ip": r.ip,
                "success": r.success,
                "error_message": r.error_message,
                "connection_time_ms": r.connection_time_ms,
                "resolution": list(r.resolution),
                "actual_fps": r.actual_fps,
                "latency_ms": r.latency_ms,
                "brightness": r.brightness,
                "contrast": r.contrast,
                "sharpness": r.sharpness,
                "noise_level": r.noise_level,
                "stability_score": r.stability_score,
                "test_duration_seconds": r.test_duration_seconds
            })
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 测试报告已保存: {output_file}")
        
        return output_file
    
    def print_summary(self, batch_result: BatchTestResult):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📊 摄像头连接测试报告")
        print("=" * 60)
        
        print(f"\n📈 总体情况:")
        print(f"  总摄像头数: {batch_result.total_cameras}")
        print(f"  成功: {batch_result.success_count} ({batch_result.success_rate:.1f}%)")
        print(f"  失败: {batch_result.failed_count}")
        print(f"  测试耗时: {batch_result.total_duration_seconds:.1f}秒")
        
        # 成功的摄像头统计
        success_results = [r for r in batch_result.results if r.success]
        if success_results:
            print(f"\n✅ 成功连接的摄像头:")
            avg_fps = np.mean([r.actual_fps for r in success_results])
            avg_latency = np.mean([r.latency_ms for r in success_results])
            avg_stability = np.mean([r.stability_score for r in success_results])
            
            print(f"  平均帧率: {avg_fps:.1f} fps")
            print(f"  平均延迟: {avg_latency:.1f} ms")
            print(f"  平均稳定性: {avg_stability:.1f}%")
        
        # 失败的摄像头
        failed_results = [r for r in batch_result.results if not r.success]
        if failed_results:
            print(f"\n❌ 失败的摄像头:")
            for r in failed_results:
                print(f"  - ID {r.camera_id} ({r.ip}): {r.error_message}")
        
        print("\n" + "=" * 60)


def quick_test_camera(
    ip: str,
    username: str = "admin",
    password: str = "",
    port: int = 554,
    channel: int = 1
) -> ConnectionTestResult:
    """
    快速测试单个摄像头
    
    Args:
        ip: 摄像头IP
        username: 用户名
        password: 密码
        port: 端口
        channel: 通道
        
    Returns:
        测试结果
    """
    config = HikvisionCameraConfig(
        camera_id=1,
        workstation_id=1,
        position="test",
        ip=ip,
        port=port,
        username=username,
        password=password,
        channel=channel,
        stream_type="sub"  # 使用子码流减少带宽
    )
    
    tester = HikvisionCameraTester(test_duration=3.0)
    return tester.test_connection(config)


# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="海康威视摄像头连接测试工具")
    parser.add_argument("--ip", type=str, help="摄像头IP地址")
    parser.add_argument("--username", type=str, default="admin", help="用户名")
    parser.add_argument("--password", type=str, default="", help="密码")
    parser.add_argument("--port", type=int, default=554, help="RTSP端口")
    parser.add_argument("--channel", type=int, default=1, help="通道号")
    parser.add_argument("--duration", type=float, default=5.0, help="测试时长（秒）")
    
    args = parser.parse_args()
    
    if args.ip:
        # 单摄像头测试
        print(f"\n🔍 测试摄像头: {args.ip}")
        result = quick_test_camera(
            ip=args.ip,
            username=args.username,
            password=args.password,
            port=args.port,
            channel=args.channel
        )
        
        if result.success:
            print(f"\n✅ 测试成功!")
            print(f"  分辨率: {result.resolution[0]}x{result.resolution[1]}")
            print(f"  帧率: {result.actual_fps:.1f} fps")
            print(f"  延迟: {result.latency_ms:.1f} ms")
            print(f"  亮度: {result.brightness:.1f}")
            print(f"  对比度: {result.contrast:.1f}")
            print(f"  清晰度: {result.sharpness:.1f}")
            print(f"  稳定性: {result.stability_score:.1f}%")
        else:
            print(f"\n❌ 测试失败: {result.error_message}")
    else:
        print("请使用 --ip 参数指定摄像头IP地址")
        print("示例: python camera_tester.py --ip 192.168.1.64 --password your_password")

