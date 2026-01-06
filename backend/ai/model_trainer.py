"""
枪械动作模型训练模块
收集训练数据、训练自定义动作识别模型
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import queue
import time
import json
import os
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionLabel(Enum):
    """动作标签"""
    IDLE = "idle"                           # 空闲
    READY_STANCE = "ready_stance"           # 准备姿势
    DRAW_GUN = "draw_gun"                   # 拔枪
    AIM = "aim"                             # 瞄准
    FIRE = "fire"                           # 射击
    RELOAD = "reload"                       # 装弹
    HOLSTER = "holster"                     # 收枪
    
    # 错误动作
    WRONG_GRIP = "wrong_grip"               # 握枪错误
    FINGER_ON_TRIGGER = "finger_on_trigger" # 手指误触扳机
    MUZZLE_UNSAFE = "muzzle_unsafe"         # 枪口不安全
    UNSTABLE_STANCE = "unstable_stance"     # 站姿不稳


@dataclass
class TrainingSample:
    """训练样本"""
    sample_id: str
    timestamp: datetime
    
    # 图像数据
    frame: Optional[np.ndarray] = None
    frame_path: str = ""
    
    # 姿态数据
    keypoints: List[Dict] = field(default_factory=list)
    
    # 标签
    action_label: ActionLabel = ActionLabel.IDLE
    quality_score: float = 0.0  # 动作质量分 0-100
    
    # 元数据
    student_id: Optional[int] = None
    workstation_id: Optional[int] = None
    instructor_id: Optional[int] = None
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "timestamp": self.timestamp.isoformat(),
            "frame_path": self.frame_path,
            "keypoints": self.keypoints,
            "action_label": self.action_label.value,
            "quality_score": self.quality_score,
            "student_id": self.student_id,
            "workstation_id": self.workstation_id,
            "instructor_id": self.instructor_id,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingSample":
        return cls(
            sample_id=data["sample_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            frame_path=data.get("frame_path", ""),
            keypoints=data.get("keypoints", []),
            action_label=ActionLabel(data["action_label"]),
            quality_score=data.get("quality_score", 0.0),
            student_id=data.get("student_id"),
            workstation_id=data.get("workstation_id"),
            instructor_id=data.get("instructor_id"),
            notes=data.get("notes", "")
        )


@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    created_at: datetime
    updated_at: datetime
    total_samples: int
    label_distribution: Dict[str, int]
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0


class DataCollector:
    """训练数据收集器"""
    
    def __init__(
        self,
        output_dir: str = "training_data",
        pose_detector = None
    ):
        """
        Args:
            output_dir: 数据输出目录
            pose_detector: 姿态检测器（可选）
        """
        self.output_dir = output_dir
        self.pose_detector = pose_detector
        
        # 创建目录结构
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # 样本计数
        self.sample_count = 0
        self.samples: List[TrainingSample] = []
        
        # 收集状态
        self.is_collecting = False
        self.current_label = ActionLabel.IDLE
        self.current_quality = 80.0
    
    def set_label(self, label: ActionLabel, quality: float = 80.0):
        """设置当前标签"""
        self.current_label = label
        self.current_quality = quality
        logger.info(f"📝 标签设置为: {label.value}, 质量分: {quality}")
    
    def collect_frame(
        self,
        frame: np.ndarray,
        keypoints: Optional[List[Dict]] = None,
        student_id: Optional[int] = None,
        workstation_id: Optional[int] = None,
        notes: str = ""
    ) -> TrainingSample:
        """
        收集一帧作为训练样本
        
        Args:
            frame: 图像帧
            keypoints: 姿态关键点（可选，如果有pose_detector会自动检测）
            student_id: 学员ID
            workstation_id: 工位ID
            notes: 备注
            
        Returns:
            训练样本
        """
        self.sample_count += 1
        sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.sample_count:06d}"
        
        # 保存图像
        frame_filename = f"{sample_id}.jpg"
        frame_path = os.path.join(self.images_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        
        # 检测姿态（如果提供了检测器且没有关键点）
        if keypoints is None and self.pose_detector:
            keypoints = self.pose_detector.detect(frame)
            if keypoints:
                keypoints = keypoints.to_dict() if hasattr(keypoints, 'to_dict') else []
        
        # 创建样本
        sample = TrainingSample(
            sample_id=sample_id,
            timestamp=datetime.now(),
            frame_path=frame_path,
            keypoints=keypoints or [],
            action_label=self.current_label,
            quality_score=self.current_quality,
            student_id=student_id,
            workstation_id=workstation_id,
            notes=notes
        )
        
        self.samples.append(sample)
        
        # 保存标签
        label_path = os.path.join(self.labels_dir, f"{sample_id}.json")
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(sample.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.debug(f"📸 收集样本: {sample_id}, 标签: {self.current_label.value}")
        
        return sample
    
    def start_auto_collect(
        self,
        camera,
        interval: float = 0.5,
        duration: Optional[float] = None
    ):
        """
        自动收集模式
        
        Args:
            camera: 摄像头实例
            interval: 采集间隔（秒）
            duration: 持续时间（秒），None表示持续到手动停止
        """
        self.is_collecting = True
        start_time = time.time()
        
        logger.info(f"🎬 开始自动收集, 间隔: {interval}秒")
        
        while self.is_collecting:
            if duration and (time.time() - start_time) > duration:
                break
            
            frame = camera.get_frame()
            if frame is not None:
                if hasattr(frame, 'frame'):
                    frame = frame.frame
                self.collect_frame(frame)
            
            time.sleep(interval)
        
        logger.info(f"⏹️ 自动收集结束, 共收集 {len(self.samples)} 个样本")
    
    def stop_auto_collect(self):
        """停止自动收集"""
        self.is_collecting = False
    
    def get_dataset_info(self) -> DatasetInfo:
        """获取数据集信息"""
        label_dist = {}
        for sample in self.samples:
            label = sample.action_label.value
            label_dist[label] = label_dist.get(label, 0) + 1
        
        return DatasetInfo(
            name=os.path.basename(self.output_dir),
            created_at=self.samples[0].timestamp if self.samples else datetime.now(),
            updated_at=self.samples[-1].timestamp if self.samples else datetime.now(),
            total_samples=len(self.samples),
            label_distribution=label_dist
        )
    
    def export_dataset(self, output_file: str = "dataset.json"):
        """导出数据集"""
        dataset = {
            "info": {
                "name": os.path.basename(self.output_dir),
                "created_at": datetime.now().isoformat(),
                "total_samples": len(self.samples)
            },
            "samples": [s.to_dict() for s in self.samples]
        }
        
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📦 数据集已导出: {output_path}")
        return output_path
    
    def load_dataset(self, dataset_file: str):
        """加载数据集"""
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.samples = [TrainingSample.from_dict(s) for s in dataset["samples"]]
        logger.info(f"📂 数据集已加载: {len(self.samples)} 个样本")


class ActionClassifierTrainer:
    """动作分类器训练器"""
    
    def __init__(
        self,
        model_type: str = "mlp",  # mlp, lstm, transformer
        input_dim: int = 33 * 4,  # 33个关键点 × 4个值(x,y,z,vis)
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = len(ActionLabel)
    ):
        """
        Args:
            model_type: 模型类型
            input_dim: 输入维度
            hidden_dims: 隐藏层维度
            num_classes: 类别数
        """
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        self.model = None
        self.label_encoder = {label.value: i for i, label in enumerate(ActionLabel)}
        self.label_decoder = {i: label.value for i, label in enumerate(ActionLabel)}
        
        # 训练历史
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def prepare_data(
        self,
        samples: List[TrainingSample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            samples: 样本列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # 提取特征和标签
        X = []
        y = []
        
        for sample in samples:
            if not sample.keypoints:
                continue
            
            # 展平关键点
            features = []
            for kp in sample.keypoints[:33]:  # 最多33个关键点
                if isinstance(kp, dict):
                    features.extend([
                        kp.get('x', 0),
                        kp.get('y', 0),
                        kp.get('z', 0),
                        kp.get('visibility', 0)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
            
            # 填充到固定维度
            while len(features) < self.input_dim:
                features.append(0)
            features = features[:self.input_dim]
            
            X.append(features)
            y.append(self.label_encoder[sample.action_label.value])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 划分数据集
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"📊 数据准备完成: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def build_model(self):
        """构建模型（使用numpy实现简单MLP）"""
        # 初始化权重
        np.random.seed(42)
        
        layers = [self.input_dim] + self.hidden_dims + [self.num_classes]
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # Xavier初始化
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        logger.info(f"🏗️ 模型构建完成: {layers}")
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """前向传播"""
        activations = [X]
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.softmax(z)
            
            activations.append(a)
        
        return activations[-1], activations
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算交叉熵损失"""
        n = len(y_true)
        # 避免log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(np.log(y_pred[np.arange(n), y_true])) / n
        return loss
    
    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        learning_rate: float
    ):
        """反向传播"""
        n = len(y)
        
        # 输出层梯度
        y_onehot = np.zeros((n, self.num_classes))
        y_onehot[np.arange(n), y] = 1
        
        delta = activations[-1] - y_onehot
        
        # 反向传播
        for i in range(len(self.weights) - 1, -1, -1):
            dw = activations[i].T @ delta / n
            db = np.mean(delta, axis=0)
            
            # 更新权重
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_derivative(activations[i])
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            early_stopping_patience: 早停耐心值
        """
        if self.weights is None:
            self.build_model()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"🚀 开始训练: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        for epoch in range(epochs):
            # 打乱训练数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批量训练
            train_losses = []
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred, activations = self.forward(X_batch)
                loss = self.compute_loss(y_pred, y_batch)
                train_losses.append(loss)
                
                # 反向传播
                self.backward(X_batch, y_batch, activations, learning_rate)
            
            # 计算训练指标
            train_loss = np.mean(train_losses)
            y_train_pred, _ = self.forward(X_train)
            train_acc = np.mean(np.argmax(y_train_pred, axis=1) == y_train)
            
            # 计算验证指标
            y_val_pred, _ = self.forward(X_val)
            val_loss = self.compute_loss(y_val_pred, y_val)
            val_acc = np.mean(np.argmax(y_val_pred, axis=1) == y_val)
            
            # 记录历史
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_acc"].append(val_acc)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                    f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
                )
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"🛑 早停于 epoch {epoch+1}")
                    # 恢复最佳模型
                    self.weights = self.best_weights
                    self.biases = self.best_biases
                    break
        
        logger.info(f"✅ 训练完成! 最佳验证损失: {best_val_loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        y_pred, _ = self.forward(X)
        return y_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型"""
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        
        # 计算每类准确率
        class_accuracy = {}
        for label_name, label_idx in self.label_encoder.items():
            mask = y == label_idx
            if np.sum(mask) > 0:
                class_accuracy[label_name] = np.mean(y_pred[mask] == y[mask])
        
        return {
            "accuracy": accuracy,
            "class_accuracy": class_accuracy
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "training_history": self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"💾 模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data["model_type"]
        self.input_dim = model_data["input_dim"]
        self.hidden_dims = model_data["hidden_dims"]
        self.num_classes = model_data["num_classes"]
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]
        self.label_encoder = model_data["label_encoder"]
        self.label_decoder = model_data["label_decoder"]
        self.training_history = model_data.get("training_history", {})
        
        logger.info(f"📂 模型已加载: {filepath}")


# 使用示例
if __name__ == "__main__":
    # 创建数据收集器
    collector = DataCollector(output_dir="training_data")
    
    # 模拟收集一些数据
    for i in range(100):
        # 模拟帧
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 模拟关键点
        keypoints = [
            {"x": np.random.rand(), "y": np.random.rand(), "z": np.random.rand() * 0.1, "visibility": 0.9}
            for _ in range(33)
        ]
        
        # 设置标签
        labels = list(ActionLabel)
        collector.set_label(labels[i % len(labels)], quality=70 + np.random.rand() * 30)
        
        # 收集样本
        collector.collect_frame(frame, keypoints=keypoints)
    
    # 导出数据集
    collector.export_dataset()
    
    # 训练模型
    trainer = ActionClassifierTrainer()
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data(collector.samples)
    
    trainer.build_model()
    trainer.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # 评估
    eval_result = trainer.evaluate(X_test, y_test)
    print(f"\n📊 测试集准确率: {eval_result['accuracy']:.4f}")
    
    # 保存模型
    trainer.save_model("action_classifier.pkl")

