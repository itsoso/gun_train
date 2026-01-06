#!/usr/bin/env python3
"""
动作识别演示程序
使用本地摄像头测试动作识别效果
"""

import cv2
import numpy as np
import sys
import os
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ai.pose_detector import PoseDetector, PoseKeypoints, AngleCalculator
from backend.ai.action_analyzer import (
    ActionAnalyzer, 
    ActionAnalysisResult,
    WarningLevel
)


class ActionAnalysisDemo:
    """动作分析演示程序"""
    
    # 错误类型的中文描述
    ERROR_DESCRIPTIONS = {
        "elbow_angle_too_small": "肘部角度过小",
        "elbow_angle_too_large": "肘部角度过大",
        "left_arm_insufficient_support": "左手支撑不足",
        "shoulder_not_level": "双肩不平衡",
        "arm_overextended": "手臂过度伸展",
        "finger_on_trigger": "⚠️ 手指在扳机上",
        "finger_not_extended": "手指未伸直",
        "head_position_low": "头部位置过低",
        "head_tilted": "头部倾斜",
        "arm_not_aligned": "手臂未对齐",
        "hand_shaking": "手部抖动严重",
        "hand_slight_shaking": "手部轻微抖动",
        "body_unstable": "身体重心不稳"
    }
    
    # 改进建议
    IMPROVEMENT_TIPS = {
        "elbow_angle_too_small": "放松手臂，自然弯曲肘部",
        "elbow_angle_too_large": "适当弯曲手臂增加稳定性",
        "left_arm_insufficient_support": "左手有力托住枪身",
        "shoulder_not_level": "保持双肩水平放松",
        "arm_overextended": "不要把手伸得太直",
        "finger_on_trigger": "非射击时手指必须离开扳机！",
        "finger_not_extended": "食指沿枪身伸直放置",
        "head_position_low": "抬头，眼睛与准星对齐",
        "head_tilted": "保持头部正直",
        "arm_not_aligned": "正面朝向目标",
        "hand_shaking": "深呼吸放松，加强练习",
        "hand_slight_shaking": "保持放松，控制呼吸",
        "body_unstable": "双脚与肩同宽，重心前倾"
    }
    
    def __init__(self, camera_source=0):
        """
        Args:
            camera_source: 摄像头源（0为默认摄像头，或RTSP URL）
        """
        self.camera_source = camera_source
        self.pose_detector = None
        self.action_analyzer = None
        self.cap = None
        
        # 统计信息
        self.frame_count = 0
        self.analysis_count = 0
        self.pass_count = 0
        self.score_history = []
        self.consecutive_passes = 0
        
        # 界面设置
        self.window_name = "🎯 动作识别分析系统"
        self.show_skeleton = True
        self.show_details = True
        
    def initialize(self) -> bool:
        """初始化"""
        print("=" * 60)
        print("   🎯 智能枪械训练 - 动作识别演示程序")
        print("=" * 60)
        print()
        
        # 初始化AI模块
        print("🤖 初始化AI模块...")
        try:
            self.pose_detector = PoseDetector(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            self.action_analyzer = ActionAnalyzer()
            print("   ✅ AI模块初始化成功")
        except Exception as e:
            print(f"   ❌ AI模块初始化失败: {e}")
            return False
        
        # 打开摄像头
        print(f"📹 连接摄像头: {self.camera_source}")
        try:
            if isinstance(self.camera_source, str) and self.camera_source.startswith("rtsp"):
                # RTSP流
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
            else:
                # 本地摄像头
                self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                raise ConnectionError("无法打开摄像头")
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            ret, frame = self.cap.read()
            if not ret:
                raise ConnectionError("无法读取视频帧")
            
            h, w = frame.shape[:2]
            print(f"   ✅ 摄像头已连接: {w}x{h}")
            
        except Exception as e:
            print(f"   ❌ 摄像头连接失败: {e}")
            return False
        
        print()
        print("📋 使用说明:")
        print("   - 站在摄像头前，模拟持枪姿势")
        print("   - 系统会实时分析你的动作规范性")
        print("   - 按 'S' 切换骨骼显示")
        print("   - 按 'D' 切换详细信息")
        print("   - 按 'R' 重置统计")
        print("   - 按 'Q' 退出程序")
        print()
        print("▶️  按任意键开始...")
        cv2.waitKey(0)
        
        return True
    
    def run(self):
        """运行演示"""
        if not self.initialize():
            return
        
        print("🎯 开始动作分析...")
        start_time = time.time()
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️ 读取帧失败，尝试重连...")
                    time.sleep(1)
                    continue
                
                self.frame_count += 1
                fps_counter += 1
                
                # 计算FPS
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    current_fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                
                # 姿态检测
                keypoints = self.pose_detector.detect(frame)
                
                # 准备显示帧
                display_frame = frame.copy()
                
                if keypoints:
                    # 绘制骨骼
                    if self.show_skeleton:
                        display_frame = self.pose_detector.draw_landmarks(frame, keypoints)
                    
                    # 动作分析
                    result = self.action_analyzer.analyze(keypoints)
                    self.analysis_count += 1
                    
                    # 更新统计
                    self.score_history.append(result.overall_score)
                    if len(self.score_history) > 100:
                        self.score_history.pop(0)
                    
                    if result.is_qualified:
                        self.pass_count += 1
                        self.consecutive_passes += 1
                    else:
                        self.consecutive_passes = 0
                    
                    # 绘制分析结果
                    self._draw_analysis_result(display_frame, result)
                    
                else:
                    # 未检测到人体
                    cv2.putText(
                        display_frame,
                        "未检测到人体姿态",
                        (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )
                
                # 绘制顶部信息栏
                self._draw_info_bar(display_frame, current_fps)
                
                # 显示
                cv2.imshow(self.window_name, display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self.show_skeleton = not self.show_skeleton
                    print(f"骨骼显示: {'开' if self.show_skeleton else '关'}")
                elif key == ord('d') or key == ord('D'):
                    self.show_details = not self.show_details
                    print(f"详细信息: {'开' if self.show_details else '关'}")
                elif key == ord('r') or key == ord('R'):
                    self._reset_stats()
                    print("统计已重置")
                    
        except KeyboardInterrupt:
            print("\n⏹️ 用户中断")
        
        finally:
            self._cleanup()
            self._print_summary()
    
    def _draw_analysis_result(self, frame: np.ndarray, result: ActionAnalysisResult):
        """绘制分析结果"""
        h, w = frame.shape[:2]
        
        # 背景面板
        panel_width = 350
        panel_height = 300 if self.show_details else 150
        cv2.rectangle(frame, (10, 60), (10 + panel_width, 60 + panel_height), 
                     (30, 30, 30), -1)
        cv2.rectangle(frame, (10, 60), (10 + panel_width, 60 + panel_height), 
                     (100, 100, 100), 2)
        
        # 总分
        y = 95
        score_color = (0, 255, 0) if result.is_qualified else (0, 0, 255)
        cv2.putText(frame, f"综合得分: {result.overall_score:.1f}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
        
        # 状态标签
        status = "合格 ✓" if result.is_qualified else "需改进 ✗"
        cv2.putText(frame, status, (250, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # 连续通过
        if self.consecutive_passes >= 3:
            cv2.putText(frame, f"连续达标: {self.consecutive_passes}", 
                       (20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if self.show_details:
            # 分项得分
            y = 145
            scores = [
                ("持枪姿势", result.posture_score),
                ("扳机纪律", result.trigger_discipline_score),
                ("瞄准线", result.aim_line_score),
                ("稳定性", result.stability_score)
            ]
            
            for name, score in scores:
                color = (0, 255, 0) if score >= 80 else (0, 165, 255) if score >= 60 else (0, 0, 255)
                cv2.putText(frame, f"{name}: {score:.1f}", 
                           (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                
                # 进度条
                bar_width = int((score / 100) * 120)
                cv2.rectangle(frame, (160, y - 12), (280, y), (50, 50, 50), -1)
                cv2.rectangle(frame, (160, y - 12), (160 + bar_width, y), color, -1)
                
                y += 28
            
            # 错误提示
            y += 10
            if result.errors:
                cv2.putText(frame, "需要改进:", (20, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                y += 25
                
                for error in result.errors[:3]:
                    # 错误描述
                    desc = self.ERROR_DESCRIPTIONS.get(error.error_type, error.description)
                    color = (0, 0, 255) if error.level == WarningLevel.SERIOUS else (0, 165, 255)
                    cv2.putText(frame, f"• {desc[:25]}", 
                               (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y += 20
        
        # 右侧改进建议面板
        if result.errors and self.show_details:
            tip_panel_x = w - 320
            cv2.rectangle(frame, (tip_panel_x, 60), (w - 10, 200), 
                         (30, 30, 30), -1)
            cv2.rectangle(frame, (tip_panel_x, 60), (w - 10, 200), 
                         (100, 100, 100), 2)
            
            cv2.putText(frame, "💡 改进建议", (tip_panel_x + 10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            y = 110
            seen_tips = set()
            for error in result.errors[:3]:
                tip = self.IMPROVEMENT_TIPS.get(error.error_type)
                if tip and tip not in seen_tips:
                    seen_tips.add(tip)
                    # 折行显示
                    if len(tip) > 18:
                        cv2.putText(frame, f"• {tip[:18]}", 
                                   (tip_panel_x + 15, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                        y += 18
                        cv2.putText(frame, f"  {tip[18:]}", 
                                   (tip_panel_x + 15, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    else:
                        cv2.putText(frame, f"• {tip}", 
                                   (tip_panel_x + 15, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    y += 25
    
    def _draw_info_bar(self, frame: np.ndarray, fps: float):
        """绘制顶部信息栏"""
        h, w = frame.shape[:2]
        
        # 背景
        cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
        
        # 标题
        cv2.putText(frame, "🎯 动作识别分析系统", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 统计信息
        avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
        pass_rate = (self.pass_count / self.analysis_count * 100) if self.analysis_count > 0 else 0
        
        info_text = f"FPS:{fps:.0f} | 分析:{self.analysis_count} | 平均:{avg_score:.1f} | 通过率:{pass_rate:.0f}%"
        cv2.putText(frame, info_text, (w - 400, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 达标提示
        if self.consecutive_passes >= 5:
            cv2.putText(frame, "🎉 已达到实弹训练标准!", (w//2 - 150, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _reset_stats(self):
        """重置统计"""
        self.analysis_count = 0
        self.pass_count = 0
        self.score_history = []
        self.consecutive_passes = 0
        self.action_analyzer.reset_stability_history()
    
    def _cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        if self.pose_detector:
            self.pose_detector.close()
        cv2.destroyAllWindows()
    
    def _print_summary(self):
        """打印总结"""
        print()
        print("=" * 60)
        print("   📊 训练总结")
        print("=" * 60)
        print(f"   分析次数: {self.analysis_count}")
        print(f"   通过次数: {self.pass_count}")
        
        if self.analysis_count > 0:
            pass_rate = self.pass_count / self.analysis_count * 100
            avg_score = sum(self.score_history) / len(self.score_history) if self.score_history else 0
            max_score = max(self.score_history) if self.score_history else 0
            
            print(f"   通过率: {pass_rate:.1f}%")
            print(f"   平均分: {avg_score:.1f}")
            print(f"   最高分: {max_score:.1f}")
            print(f"   最大连续通过: {self.consecutive_passes}")
        
        print("=" * 60)
        print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="动作识别演示程序")
    parser.add_argument(
        "--camera", 
        type=str, 
        default="0",
        help="摄像头源（0为默认摄像头，或RTSP URL）"
    )
    
    args = parser.parse_args()
    
    # 解析摄像头源
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)
    
    # 运行演示
    demo = ActionAnalysisDemo(camera_source)
    demo.run()


if __name__ == "__main__":
    main()

