"""
性能优化模块
GPU批处理、多线程优化、内存管理
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import deque
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 处理速度
    fps: float = 0.0
    frames_processed: int = 0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    
    # 队列状态
    input_queue_size: int = 0
    output_queue_size: int = 0
    
    # 资源使用
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # 吞吐量
    throughput_per_second: float = 0.0
    dropped_frames: int = 0
    
    # 时间统计
    uptime_seconds: float = 0.0
    start_time: Optional[datetime] = None


class FrameBatcher:
    """帧批处理器 - 将多帧打包成批次进行处理"""
    
    def __init__(
        self,
        batch_size: int = 8,
        max_wait_time: float = 0.1,  # 最大等待时间（秒）
        input_shape: Tuple[int, int, int] = (480, 640, 3)
    ):
        """
        Args:
            batch_size: 批大小
            max_wait_time: 最大等待时间
            input_shape: 输入帧形状
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.input_shape = input_shape
        
        self.frame_buffer: List[Tuple[Any, np.ndarray]] = []  # (metadata, frame)
        self.buffer_lock = threading.Lock()
        self.last_batch_time = time.time()
        
        # 统计
        self.total_batches = 0
        self.total_frames = 0
    
    def add_frame(self, frame: np.ndarray, metadata: Any = None) -> Optional[Tuple[List[Any], np.ndarray]]:
        """
        添加帧到批次缓冲区
        
        Args:
            frame: 图像帧
            metadata: 元数据
            
        Returns:
            如果批次已满或超时，返回(元数据列表, 批次数组)，否则返回None
        """
        with self.buffer_lock:
            # 调整帧大小
            if frame.shape != self.input_shape:
                frame = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
            
            self.frame_buffer.append((metadata, frame))
            
            # 检查是否需要返回批次
            current_time = time.time()
            should_return = (
                len(self.frame_buffer) >= self.batch_size or
                (len(self.frame_buffer) > 0 and 
                 current_time - self.last_batch_time >= self.max_wait_time)
            )
            
            if should_return:
                return self._flush_buffer()
            
            return None
    
    def _flush_buffer(self) -> Tuple[List[Any], np.ndarray]:
        """刷新缓冲区并返回批次"""
        metadata_list = [item[0] for item in self.frame_buffer]
        frames = np.array([item[1] for item in self.frame_buffer])
        
        self.frame_buffer.clear()
        self.last_batch_time = time.time()
        self.total_batches += 1
        self.total_frames += len(frames)
        
        return metadata_list, frames
    
    def flush(self) -> Optional[Tuple[List[Any], np.ndarray]]:
        """强制刷新缓冲区"""
        with self.buffer_lock:
            if self.frame_buffer:
                return self._flush_buffer()
            return None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_batches": self.total_batches,
            "total_frames": self.total_frames,
            "avg_batch_size": self.total_frames / max(self.total_batches, 1),
            "buffer_size": len(self.frame_buffer)
        }


class ThreadPoolManager:
    """线程池管理器"""
    
    def __init__(
        self,
        max_workers: int = None,
        thread_name_prefix: str = "worker"
    ):
        """
        Args:
            max_workers: 最大工作线程数，默认为CPU核心数
            thread_name_prefix: 线程名前缀
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_name_prefix = thread_name_prefix
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix
        )
        
        self.pending_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.task_lock = threading.Lock()
        
        logger.info(f"🔧 线程池初始化: {self.max_workers} 个工作线程")
    
    def submit(self, fn: Callable, *args, **kwargs):
        """提交任务"""
        with self.task_lock:
            self.pending_tasks += 1
        
        future = self.executor.submit(self._wrapped_fn, fn, *args, **kwargs)
        return future
    
    def _wrapped_fn(self, fn: Callable, *args, **kwargs):
        """包装函数用于统计"""
        try:
            result = fn(*args, **kwargs)
            with self.task_lock:
                self.completed_tasks += 1
                self.pending_tasks -= 1
            return result
        except Exception as e:
            with self.task_lock:
                self.failed_tasks += 1
                self.pending_tasks -= 1
            raise e
    
    def map(self, fn: Callable, iterables, timeout: float = None):
        """批量提交任务"""
        return self.executor.map(fn, iterables, timeout=timeout)
    
    def shutdown(self, wait: bool = True):
        """关闭线程池"""
        self.executor.shutdown(wait=wait)
        logger.info("🔧 线程池已关闭")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "max_workers": self.max_workers,
            "pending_tasks": self.pending_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks
        }


class AsyncFrameProcessor:
    """异步帧处理器"""
    
    def __init__(
        self,
        process_fn: Callable[[np.ndarray], Any],
        num_workers: int = 4,
        input_queue_size: int = 100,
        output_queue_size: int = 100
    ):
        """
        Args:
            process_fn: 处理函数
            num_workers: 工作线程数
            input_queue_size: 输入队列大小
            output_queue_size: 输出队列大小
        """
        self.process_fn = process_fn
        self.num_workers = num_workers
        
        self.input_queue = queue.Queue(maxsize=input_queue_size)
        self.output_queue = queue.Queue(maxsize=output_queue_size)
        
        self.workers: List[threading.Thread] = []
        self.is_running = False
        
        # 性能指标
        self.metrics = PerformanceMetrics(start_time=datetime.now())
        self.processing_times: deque = deque(maxlen=100)
    
    def start(self):
        """启动处理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.metrics.start_time = datetime.now()
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"frame_processor_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"🚀 异步帧处理器启动: {self.num_workers} 个工作线程")
    
    def stop(self):
        """停止处理器"""
        self.is_running = False
        
        # 发送停止信号
        for _ in self.workers:
            try:
                self.input_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
        
        # 等待线程结束
        for worker in self.workers:
            worker.join(timeout=2)
        
        self.workers.clear()
        logger.info("⏹️ 异步帧处理器已停止")
    
    def _worker_loop(self):
        """工作线程循环"""
        while self.is_running:
            try:
                item = self.input_queue.get(timeout=0.1)
                
                if item is None:  # 停止信号
                    break
                
                frame_id, frame, metadata = item
                
                # 处理帧
                start_time = time.time()
                try:
                    result = self.process_fn(frame)
                    success = True
                except Exception as e:
                    result = None
                    success = False
                    logger.error(f"处理帧时出错: {e}")
                
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                # 更新指标
                self.metrics.frames_processed += 1
                self.metrics.max_processing_time_ms = max(
                    self.metrics.max_processing_time_ms, processing_time
                )
                self.metrics.min_processing_time_ms = min(
                    self.metrics.min_processing_time_ms, processing_time
                )
                
                # 放入输出队列
                output_item = (frame_id, result, metadata, success, processing_time)
                try:
                    self.output_queue.put(output_item, timeout=0.1)
                except queue.Full:
                    self.metrics.dropped_frames += 1
                
            except queue.Empty:
                continue
    
    def submit(self, frame: np.ndarray, frame_id: int = None, metadata: Any = None):
        """
        提交帧进行处理
        
        Args:
            frame: 图像帧
            frame_id: 帧ID
            metadata: 元数据
        """
        if frame_id is None:
            frame_id = self.metrics.frames_processed
        
        try:
            self.input_queue.put((frame_id, frame, metadata), timeout=0.01)
        except queue.Full:
            self.metrics.dropped_frames += 1
    
    def get_result(self, timeout: float = 0.1) -> Optional[Tuple]:
        """
        获取处理结果
        
        Returns:
            (frame_id, result, metadata, success, processing_time_ms)
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        if self.processing_times:
            self.metrics.avg_processing_time_ms = np.mean(list(self.processing_times))
        
        self.metrics.input_queue_size = self.input_queue.qsize()
        self.metrics.output_queue_size = self.output_queue.qsize()
        
        if self.metrics.start_time:
            elapsed = (datetime.now() - self.metrics.start_time).total_seconds()
            self.metrics.uptime_seconds = elapsed
            if elapsed > 0:
                self.metrics.fps = self.metrics.frames_processed / elapsed
                self.metrics.throughput_per_second = self.metrics.frames_processed / elapsed
        
        return self.metrics


class GPUBatchProcessor:
    """GPU批处理器（使用OpenCV的CUDA后端）"""
    
    def __init__(
        self,
        batch_size: int = 16,
        use_gpu: bool = True
    ):
        """
        Args:
            batch_size: 批大小
            use_gpu: 是否使用GPU
        """
        self.batch_size = batch_size
        self.use_gpu = use_gpu and self._check_gpu_available()
        
        self.batcher = FrameBatcher(batch_size=batch_size)
        
        if self.use_gpu:
            logger.info("🎮 GPU加速已启用")
        else:
            logger.info("💻 使用CPU处理")
    
    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        try:
            # 检查OpenCV CUDA支持
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
        except:
            pass
        
        # 检查是否有CUDA设备
        try:
            import torch
            return torch.cuda.is_available()
        except:
            pass
        
        return False
    
    def preprocess_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        批量预处理
        
        Args:
            frames: 批次帧 (N, H, W, C)
            
        Returns:
            预处理后的帧
        """
        if self.use_gpu:
            try:
                # 使用GPU进行预处理
                processed = []
                for frame in frames:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    
                    # 调整大小
                    gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))
                    
                    # 归一化
                    processed.append(gpu_resized.download())
                
                return np.array(processed)
            except Exception as e:
                logger.warning(f"GPU处理失败，回退到CPU: {e}")
        
        # CPU处理
        processed = []
        for frame in frames:
            resized = cv2.resize(frame, (640, 480))
            processed.append(resized)
        
        return np.array(processed)
    
    def process_batch(
        self,
        frames: np.ndarray,
        process_fn: Callable[[np.ndarray], Any]
    ) -> List[Any]:
        """
        批量处理
        
        Args:
            frames: 批次帧
            process_fn: 处理函数
            
        Returns:
            处理结果列表
        """
        # 预处理
        preprocessed = self.preprocess_batch(frames)
        
        # 处理
        results = []
        for frame in preprocessed:
            result = process_fn(frame)
            results.append(result)
        
        return results


class MemoryManager:
    """内存管理器"""
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        gc_threshold: float = 0.8
    ):
        """
        Args:
            max_memory_mb: 最大内存使用（MB）
            gc_threshold: GC触发阈值
        """
        self.max_memory_mb = max_memory_mb
        self.gc_threshold = gc_threshold
        
        # 对象池
        self.frame_pool: List[np.ndarray] = []
        self.pool_size = 50
        self.pool_lock = threading.Lock()
        
        # 统计
        self.allocations = 0
        self.deallocations = 0
        self.gc_runs = 0
    
    def get_frame_buffer(self, shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
        """
        从池中获取帧缓冲区
        
        Args:
            shape: 帧形状
            
        Returns:
            帧缓冲区
        """
        with self.pool_lock:
            if self.frame_pool:
                buffer = self.frame_pool.pop()
                if buffer.shape == shape:
                    self.allocations += 1
                    return buffer
            
            # 创建新缓冲区
            self.allocations += 1
            return np.zeros(shape, dtype=np.uint8)
    
    def release_frame_buffer(self, buffer: np.ndarray):
        """
        释放帧缓冲区回池
        
        Args:
            buffer: 帧缓冲区
        """
        with self.pool_lock:
            if len(self.frame_pool) < self.pool_size:
                self.frame_pool.append(buffer)
            self.deallocations += 1
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用（MB）"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def check_and_gc(self):
        """检查并执行GC"""
        current_usage = self.get_memory_usage()
        
        if current_usage > self.max_memory_mb * self.gc_threshold:
            gc.collect()
            self.gc_runs += 1
            
            new_usage = self.get_memory_usage()
            logger.info(f"🧹 GC执行: {current_usage:.1f}MB -> {new_usage:.1f}MB")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "memory_usage_mb": self.get_memory_usage(),
            "max_memory_mb": self.max_memory_mb,
            "pool_size": len(self.frame_pool),
            "allocations": self.allocations,
            "deallocations": self.deallocations,
            "gc_runs": self.gc_runs
        }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 统计窗口大小
        """
        self.window_size = window_size
        
        self.frame_times: deque = deque(maxlen=window_size)
        self.processing_times: deque = deque(maxlen=window_size)
        
        self.start_time = time.time()
        self.total_frames = 0
    
    def record_frame(self, processing_time_ms: float):
        """
        记录帧处理时间
        
        Args:
            processing_time_ms: 处理时间（毫秒）
        """
        current_time = time.time()
        self.frame_times.append(current_time)
        self.processing_times.append(processing_time_ms)
        self.total_frames += 1
    
    def get_fps(self) -> float:
        """获取实时FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_span
    
    def get_avg_processing_time(self) -> float:
        """获取平均处理时间（毫秒）"""
        if not self.processing_times:
            return 0.0
        return np.mean(list(self.processing_times))
    
    def get_p95_processing_time(self) -> float:
        """获取P95处理时间（毫秒）"""
        if not self.processing_times:
            return 0.0
        return np.percentile(list(self.processing_times), 95)
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            "fps": round(self.get_fps(), 2),
            "avg_processing_time_ms": round(self.get_avg_processing_time(), 2),
            "p95_processing_time_ms": round(self.get_p95_processing_time(), 2),
            "total_frames": self.total_frames,
            "uptime_seconds": round(time.time() - self.start_time, 1)
        }


# 工厂函数
def create_optimized_processor(
    process_fn: Callable,
    num_workers: int = 4,
    batch_size: int = 8,
    use_gpu: bool = True
) -> AsyncFrameProcessor:
    """
    创建优化的帧处理器
    
    Args:
        process_fn: 处理函数
        num_workers: 工作线程数
        batch_size: 批大小
        use_gpu: 是否使用GPU
        
    Returns:
        异步帧处理器
    """
    # 创建GPU批处理器
    gpu_processor = GPUBatchProcessor(batch_size=batch_size, use_gpu=use_gpu)
    
    # 包装处理函数
    def optimized_process(frame):
        # 这里可以添加GPU预处理
        return process_fn(frame)
    
    # 创建异步处理器
    processor = AsyncFrameProcessor(
        process_fn=optimized_process,
        num_workers=num_workers
    )
    
    return processor


# 使用示例
if __name__ == "__main__":
    # 测试异步帧处理器
    def dummy_process(frame):
        """模拟处理函数"""
        time.sleep(0.01)  # 模拟处理延迟
        return {"shape": frame.shape, "mean": np.mean(frame)}
    
    processor = AsyncFrameProcessor(
        process_fn=dummy_process,
        num_workers=4
    )
    
    processor.start()
    
    # 提交帧
    for i in range(100):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processor.submit(frame, frame_id=i)
    
    # 获取结果
    results_count = 0
    start = time.time()
    
    while results_count < 100:
        result = processor.get_result(timeout=1.0)
        if result:
            results_count += 1
    
    elapsed = time.time() - start
    
    # 打印统计
    metrics = processor.get_metrics()
    print(f"\n📊 性能统计:")
    print(f"  处理帧数: {metrics.frames_processed}")
    print(f"  FPS: {metrics.fps:.2f}")
    print(f"  平均处理时间: {metrics.avg_processing_time_ms:.2f}ms")
    print(f"  丢帧数: {metrics.dropped_frames}")
    
    processor.stop()

