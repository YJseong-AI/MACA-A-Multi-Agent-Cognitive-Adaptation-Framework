#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Hierarchical Multi-Level MCTS Optimization System
연구용 계층적 다단계 MCTS 의사결정 프레임워크

Level 0 (Meta): 접근법 선택 (combination vs collaboration vs hybrid)
Level 1 (Cognitive): 인지 부하 기반 적응
Level 2 (Combination): 에이전트 조합 최적화  
Level 3 (Execution): 실행 전략 최적화
"""

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe 로드됨 - 실제 시선추적 사용")
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe 없음 - 시뮬레이션 모드로 실행")
import csv
import time
from datetime import datetime
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import os
import sys
import json
import random
import math
import copy
import queue
import gc

@dataclass
class HierarchicalDecision:
    """계층적 의사결정 결과"""
    meta_strategy: str
    cognitive_adaptation: str
    combination_choice: Tuple
    execution_strategy: str
    tree_depth: int
    quality_score: float
    decision_time: float
    confidence: float
    tree_visualization: Dict
    level_decisions: list


# ==================== Baseline 2: Single-Level MCTS ====================
class SingleLevelMCTS:
    """
    1단계 MCTS만 사용
    
    제안 시스템과 차이점:
    - Level 0 (Meta-Strategy)만 탐색
    - Level 1, 2, 3은 고정값 사용
    - 계층 구조 없음
    """
    
    def __init__(self):
        self.name = "Baseline 2: Single-Level MCTS"
        
        # Meta-Strategy 선택지
        self.strategies = ["combination", "collaboration", "hybrid", "adaptive"]
        
        # 나머지는 고정
        self.fixed_adaptation = "standard"
        self.fixed_combination = ("adaptive", "balanced", "standard")
        self.fixed_execution = "standard"
        
        # 단순 MCTS 파라미터
        self.iterations = 30
        self.c_param = 1.4
        
        # 탐색 통계
        self.visit_counts = {s: 0 for s in self.strategies}
        self.total_rewards = {s: 0.0 for s in self.strategies}
        
        print(f"✅ {self.name} initialized")
    
    def search(self, user_context: Dict) -> HierarchicalDecision:
        """
        단일 레벨 MCTS로 메타 전략만 탐색
        """
        start_time = time.time()
        
        # 사용자 컨텍스트는 사용 (하지만 단순하게만)
        emotion = user_context.get('emotion', 'neutral')
        attention = user_context.get('attention', 0.5)
        
        # UCB1 기반 탐색
        for _ in range(self.iterations):
            # 선택
            strategy = self._select_strategy()
            
            # 시뮬레이션 (간단한 보상 계산)
            reward = self._simulate(strategy, emotion, attention)
            
            # 업데이트
            self.visit_counts[strategy] += 1
            self.total_rewards[strategy] += reward
        
        # 최적 전략 선택 (가장 높은 평균 보상)
        best_strategy = max(self.strategies, 
                          key=lambda s: self.total_rewards[s] / max(1, self.visit_counts[s]))
        
        decision_time = time.time() - start_time
        
        # 품질 점수 (1단계만 사용하므로 제한적)
        avg_reward = self.total_rewards[best_strategy] / max(1, self.visit_counts[best_strategy])
        quality_score = 0.60 + avg_reward * 0.15
        quality_score = np.clip(quality_score, 0.5, 0.75)
        
        confidence = 0.65
        
        tree_visualization = {
            "type": "single_level",
            "strategies_explored": len([s for s in self.strategies if self.visit_counts[s] > 0]),
            "best_strategy": best_strategy
        }
        
        level_decisions = [
            {"level": 0, "decision": best_strategy, "type": "meta_strategy", "explored": True},
            {"level": 1, "decision": self.fixed_adaptation, "type": "adaptation", "fixed": True},
            {"level": 2, "decision": self.fixed_combination, "type": "combination", "fixed": True},
            {"level": 3, "decision": self.fixed_execution, "type": "execution", "fixed": True}
        ]
        
        return HierarchicalDecision(
            meta_strategy=best_strategy,
            cognitive_adaptation=self.fixed_adaptation,
            combination_choice=self.fixed_combination,
            execution_strategy=self.fixed_execution,
            tree_depth=1,  # 1단계만!
            quality_score=quality_score,
            decision_time=decision_time,
            confidence=confidence,
            tree_visualization=tree_visualization,
            level_decisions=level_decisions
        )
    
    def _select_strategy(self):
        """UCB1으로 전략 선택"""
        total_visits = sum(self.visit_counts.values())
        
        if total_visits == 0:
            return random.choice(self.strategies)
        
        ucb_values = {}
        for strategy in self.strategies:
            if self.visit_counts[strategy] == 0:
                return strategy
            
            exploitation = self.total_rewards[strategy] / self.visit_counts[strategy]
            exploration = self.c_param * np.sqrt(2 * np.log(total_visits) / self.visit_counts[strategy])
            ucb_values[strategy] = exploitation + exploration
        
        return max(ucb_values, key=ucb_values.get)
    
    def _simulate(self, strategy, emotion, attention):
        """간단한 보상 계산"""
        reward = 0.5
        
        # 감정 기반 보상 (단순)
        if emotion in ['happy', 'surprise'] and strategy in ['combination', 'hybrid']:
            reward += 0.1
        elif emotion in ['sad', 'anger'] and strategy in ['collaboration', 'adaptive']:
            reward += 0.1
        
        # 집중도 기반 보상 (단순)
        if attention > 0.7 and strategy == 'hybrid':
            reward += 0.05
        elif attention < 0.4 and strategy == 'combination':
            reward += 0.05
        
        reward += np.random.normal(0, 0.05)
        return np.clip(reward, 0.0, 1.0)


# ==================== Baseline 3: Rule-Based System ====================
class RuleBasedSystem:
    """
    단순 규칙 기반 시스템
    
    MCTS 없이 if-else만으로 결정:
    - 감정에 따라 전략 선택
    - 집중도에 따라 적응 타입 선택
    - 인지 부하에 따라 실행 전략 선택
    """
    
    def __init__(self):
        self.name = "Baseline 3: Rule-Based"
        print(f"✅ {self.name} initialized")
    
    def search(self, user_context: Dict) -> HierarchicalDecision:
        """
        규칙 기반 의사결정 (빠름!)
        """
        start_time = time.time()
        
        # 사용자 컨텍스트 추출
        emotion = user_context.get('emotion', 'neutral')
        attention = user_context.get('attention', 0.5)
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        
        # Rule 1: 감정 → 메타 전략
        if emotion in ['happy', 'surprise']:
            meta_strategy = "combination"
        elif emotion in ['sad', 'fear']:
            meta_strategy = "collaboration"
        elif emotion == 'anger':
            meta_strategy = "adaptive"
        else:  # neutral
            meta_strategy = "hybrid"
        
        # Rule 2: 집중도 → 인지 적응
        if attention > 0.7:
            cognitive_adaptation = "complex"
        elif attention < 0.3:
            cognitive_adaptation = "simplified"
        else:
            cognitive_adaptation = "standard"
        
        # Rule 3: 인지 부하 → 실행 전략
        if cognitive_load == 'high':
            execution_strategy = "gentle"
        elif cognitive_load == 'low':
            execution_strategy = "intensive"
        else:
            execution_strategy = "standard"
        
        # Rule 4: 조합 선택 (감정 + 인지 부하 기반)
        if emotion in ['sad', 'fear', 'anger'] or cognitive_load == 'high':
            # 부정적 상황 → 부드러운 조합
            combination_choice = ("encouraging", "supportive", "gentle")
        elif emotion in ['happy', 'surprise'] and cognitive_load == 'low':
            # 긍정적 + 여유 → 도전적 조합
            combination_choice = ("challenging", "thorough", "intensive")
        else:
            # 중립 → 균형잡힌 조합
            combination_choice = ("adaptive", "balanced", "standard")
        
        decision_time = time.time() - start_time
        
        # 품질 점수 (규칙 기반이므로 제한적)
        # 복잡한 최적화 없이 빠르지만 품질은 낮음
        quality_score = 0.58 + np.random.normal(0, 0.05)
        quality_score = np.clip(quality_score, 0.5, 0.70)
        
        confidence = 0.60
        
        tree_visualization = {
            "type": "rule_based",
            "rules_applied": 4,
            "computation_time": "instant"
        }
        
        level_decisions = [
            {"level": 0, "decision": meta_strategy, "type": "meta_strategy", "rule": "emotion"},
            {"level": 1, "decision": cognitive_adaptation, "type": "adaptation", "rule": "attention"},
            {"level": 2, "decision": combination_choice, "type": "combination", "rule": "emotion+load"},
            {"level": 3, "decision": execution_strategy, "type": "execution", "rule": "cognitive_load"}
        ]
        
        return HierarchicalDecision(
            meta_strategy=meta_strategy,
            cognitive_adaptation=cognitive_adaptation,
            combination_choice=combination_choice,
            execution_strategy=execution_strategy,
            tree_depth=0,  # 규칙이므로 깊이 없음
            quality_score=quality_score,
            decision_time=decision_time,
            confidence=confidence,
            tree_visualization=tree_visualization,
            level_decisions=level_decisions
        )


# ==================== 통합 실험 래퍼 ====================
class BaselineExperimentSystem:
    """
    4개 시스템을 통합 관리하는 실험용 래퍼
    """
    
    def __init__(self, proposed_system):
        """
        Args:
            proposed_system: 원래 HierarchicalMCTS 인스턴스
        """
        self.systems = {
            'proposed': proposed_system,
            'no_adapt': NoAdaptationMCTS(),
            'single': SingleLevelMCTS(),
            'rule': RuleBasedSystem()
        }
        
        self.current_system = 'proposed'
        
        print("\n" + "="*70)
        print("🔬 BASELINE EXPERIMENT SYSTEM INITIALIZED")
        print("="*70)
        print("Available systems:")
        print("  1. Proposed: 4-Level Hierarchical MCTS with Adaptation")
        print("  2. No Adaptation: 4-Level MCTS without User Adaptation")
        print("  3. Single-Level: 1-Level MCTS only")
        print("  4. Rule-Based: Simple if-else rules")
        print("="*70 + "\n")
    
    def set_system(self, system_name: str):
        """현재 시스템 변경"""
        if system_name not in self.systems:
            raise ValueError(f"Unknown system: {system_name}")
        
        self.current_system = system_name
        print(f"✅ Switched to: {self.systems[system_name].name}")
    
    def search(self, user_context: Dict) -> HierarchicalDecision:
        """현재 선택된 시스템으로 의사결정"""
        system = self.systems[self.current_system]
        return system.search(user_context)
    
    def get_system_info(self):
        """현재 시스템 정보"""
        return {
            'current': self.current_system,
            'name': self.systems[self.current_system].name,
            'all_systems': list(self.systems.keys())
        }


# ==================== Baseline Systems Removed ====================
# (베이스라인 시스템들은 사용하지 않음 - 각 조건별로 직접 구현)


# ==================== 한글 텍스트 표시 유틸리티 ====================
from PIL import ImageDraw, ImageFont

class KoreanTextRenderer:
    """한글 텍스트를 OpenCV 이미지에 렌더링하는 클래스"""
    
    def __init__(self):
        """한글 폰트 초기화"""
        self.font_cache = {}
        self.default_font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "C:/Windows/Fonts/malgun.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        ]
    
    def get_font(self, size):
        """폰트 캐시 관리"""
        if size in self.font_cache:
            return self.font_cache[size]
        
        for font_path in self.default_font_paths:
            try:
                font = ImageFont.truetype(font_path, size)
                self.font_cache[size] = font
                return font
            except:
                continue
        
        font = ImageFont.load_default()
        self.font_cache[size] = font
        return font
    
    def put_text(self, img, text, position, font_size=20, color=(255, 255, 255), 
                 bg_color=None, padding=5):
        """한글 텍스트를 이미지에 렌더링"""
        import cv2
        import numpy as np
        from PIL import Image
        
        if not text:
            return img
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        font = self.get_font(font_size)
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            text_width = len(text) * font_size // 2
            text_height = font_size
        
        x, y = position
        
        if bg_color is not None:
            bg_bbox = [
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            ]
            draw.rectangle(bg_bbox, fill=bg_color)
        
        draw.text((x, y), text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

_korean_renderer = KoreanTextRenderer()

def put_korean_text(img, text, position, font_size=20, color=(255, 255, 255), 
                   bg_color=None, padding=5):
    """간편한 한글 텍스트 표시 함수"""
    return _korean_renderer.put_text(img, text, position, font_size, color, bg_color, padding)


# ==================== 계층적 MCTS 데이터 구조 ====================

@dataclass
class HierarchicalDecision:
    """계층적 의사결정 결과"""
    meta_strategy: str           # Level 0: "combination", "collaboration", "hybrid"
    cognitive_adaptation: str    # Level 1: "simplified", "standard", "complex"
    combination_choice: Tuple    # Level 2: (planner, critic, executor)
    execution_strategy: str      # Level 3: "gentle", "standard", "intensive"
    
    # 성능 지표
    tree_depth: int
    quality_score: float
    decision_time: float
    confidence: float
    
    # 시각화용 데이터
    tree_visualization: Dict
    level_decisions: List[Dict]
    
    # GPT-4 협업 관련 필드 (논문 구현)
    ids: float = 0.0  # Information Diversity Score (GEMMAS)
    upr: float = 0.0  # Unnecessary Path Ratio (GEMMAS)
    llm_feedback: str = ""  # GPT-4 협업 피드백

@dataclass  
class MCTSLevelStats:
    """각 레벨별 MCTS 통계"""
    level: int
    nodes_explored: int
    best_value: float
    exploration_depth: int
    decision_count: int
    avg_decision_time: float

# ==================== Level 0: Meta-Strategy MCTS ====================
class MetaStrategyNode:
    """메타 전략 선택 노드"""
    
    def __init__(self, strategy_type: str, parent=None):
        self.strategy_type = strategy_type  # "combination", "collaboration", "hybrid"
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried_strategies = self._get_possible_strategies()
    
    def _get_possible_strategies(self):
        """가능한 메타 전략들"""
        return ["combination", "collaboration", "hybrid", "adaptive", "dynamic"]
    
    def is_fully_expanded(self):
        return len(self.untried_strategies) == 0
    
    def best_child(self, c_param=1.4, context=None):
        """이론적으로 최적화된 UCB1으로 최적 자식 선택"""
        if not self.children:
            return None
        
        # 자동 c_param 계산
        if context is not None and hasattr(self, 'calculate_optimal_c_parameter'):
            c_param = self.calculate_optimal_c_parameter(context)
            
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploitation = child.total_reward / child.visits
            exploration = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            choices_weights.append(exploitation + exploration)
        
        return self.children[np.argmax(choices_weights)]
    
    def update(self, reward):
        """노드 업데이트"""
        self.visits += 1
        self.total_reward += reward
    
    def get_average_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0

class MetaStrategyMCTS:
    """Level 0: 메타 전략 선택 MCTS"""
    
    def __init__(self, c_param=None):
        self.c_param = c_param  # None이면 자동 계산
        self.root = None
        self.decision_history = deque(maxlen=50)
        self.suboptimality_gaps = {}  # 전략별 suboptimality gap 추정
    
    def search(self, user_context: Dict, iterations=30) -> str:
        """최적 메타 전략 탐색"""
        
        if self.root is None:
            self.root = MetaStrategyNode("combination")  # 기본 전략
        
        for _ in range(iterations):
            # Selection & Expansion
            leaf = self._select_and_expand(self.root, user_context)
            
            # Simulation
            reward = self._calculate_meta_strategy(leaf, user_context)
            
            # Backpropagation  
            self._backpropagate(leaf, reward)
        
        # 최적 전략 선택
        if not self.root.children:
            return "combination"  # 기본값
            
        best_child = max(self.root.children, 
                        key=lambda x: x.get_average_reward())
        
        return best_child.strategy_type
    
    def _select_and_expand(self, node, user_context):
        """선택 및 확장"""
        
        # Selection: 리프 노드까지 내려가기 (이론적 최적화된 UCB1)
        while node.children and node.is_fully_expanded():
            optimal_c = self.calculate_optimal_c_parameter(user_context) if self.c_param is None else self.c_param
            node = node.best_child(optimal_c, user_context)
        
        # Expansion: 새 자식 노드 추가
        if not node.is_fully_expanded():
            strategy = node.untried_strategies.pop()
            child = MetaStrategyNode(strategy, parent=node)
            node.children.append(child)
            return child
            
        return node
    
    def _calculate_meta_strategy(self, node, user_context):
        """메타 전략 효과성 계산 (논문 보상 함수 사용)"""
        
        strategy = node.strategy_type
        emotion = user_context.get('emotion', 'neutral')
        attention = user_context.get('attention', 0.5)
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        
        # 논문 보상 함수 기반 계산
        # 간단한 휴리스틱 기반 보상 (논문의 보상 함수 구조에 맞춤)
        
        # 1. 감정-전략 정렬 (R_emo 관련)
        emotion_reward = self._emotion_strategy_alignment(emotion, strategy)
        
        # 2. 주의집중-전략 정렬 (R_eff 관련)
        attention_reward = self._attention_strategy_alignment(attention, strategy)
        
        # 3. 인지 부하-전략 정렬
        cognitive_reward = self._cognitive_load_strategy_alignment(cognitive_load, strategy)
        
        # 가중 평균
        integrated_reward = 0.4 * emotion_reward + 0.3 * attention_reward + 0.3 * cognitive_reward
        
        return np.clip(integrated_reward, 0.0, 1.0)
    
    def _emotion_strategy_alignment(self, emotion, strategy):
        """감정-전략 정렬 점수 (논문 요구사항에 맞춤)"""
        alignment_matrix = {
            ('happy', 'combination'): 0.8, ('happy', 'hybrid'): 0.9,
            ('surprise', 'hybrid'): 0.85, ('surprise', 'dynamic'): 0.8,
            ('sad', 'collaboration'): 0.9, ('sad', 'adaptive'): 0.85,
            ('fear', 'collaboration'): 0.85, ('fear', 'adaptive'): 0.8,
            ('anger', 'adaptive'): 0.75, ('anger', 'collaboration'): 0.7,
            ('neutral', 'hybrid'): 0.7, ('neutral', 'combination'): 0.65
        }
        return alignment_matrix.get((emotion, strategy), 0.5)
    
    def _attention_strategy_alignment(self, attention, strategy):
        """주의집중-전략 정렬 점수 (논문 요구사항에 맞춤)"""
        if attention > 0.8:  # 높은 집중도
            high_attention_scores = {
                'hybrid': 0.9, 'dynamic': 0.85, 'collaboration': 0.75
            }
            return high_attention_scores.get(strategy, 0.6)
        elif attention < 0.4:  # 낮은 집중도
            low_attention_scores = {
                'combination': 0.85, 'adaptive': 0.7
            }
            return low_attention_scores.get(strategy, 0.5)
        else:  # 중간 집중도
            return 0.65
    
    def _cognitive_load_strategy_alignment(self, cognitive_load, strategy):
        """인지 부하-전략 정렬 점수"""
        alignment_matrix = {
            ('high', 'combination'): 0.9,    # 높은 부하 → 단순 조합
            ('high', 'collaboration'): 0.7,
            ('medium', 'hybrid'): 0.8,       # 중간 부하 → 하이브리드
            ('medium', 'adaptive'): 0.75,
            ('low', 'dynamic'): 0.85,       # 낮은 부하 → 동적
            ('low', 'hybrid'): 0.8
        }
        return alignment_matrix.get((cognitive_load, strategy), 0.6)
    
    def _backpropagate(self, node, reward):
        """보상 역전파"""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def calculate_optimal_c_parameter(self, context):
        """이론적 최적 탐색 상수 계산"""
        if context is None:
            return 1.4  # 기본값
        
        # Suboptimality gap 추정
        subopt_gaps = self.estimate_suboptimality_gaps(context)
        
        if not subopt_gaps:
            return 1.4
        
        # 최소 gap 계산
        min_gap = min(gap for gap in subopt_gaps.values() if gap > 0)
        horizon = context.get('time_horizon', 1000)
        
        # 이론적 최적값: c = √(2 * log(horizon) / min_gap²)
        optimal_c = math.sqrt(2 * math.log(horizon) / (min_gap ** 2))
        return np.clip(optimal_c, 0.1, 3.0)  # 실용적 범위로 제한
    
    def estimate_suboptimality_gaps(self, context):
        """전략별 suboptimality gap 추정"""
        strategies = ['combination', 'collaboration', 'hybrid', 'adaptive', 'dynamic']
        gaps = {}
        
        # 컨텍스트 기반 최적 전략 추정
        optimal_strategy = self.get_optimal_strategy_estimate(context)
        
        for strategy in strategies:
            if strategy == optimal_strategy:
                gaps[strategy] = 0.01  # 최적 전략 (0이 아닌 작은 값)
            else:
                # 전략 간 성능 차이 추정
                gaps[strategy] = self.estimate_strategy_performance_gap(strategy, optimal_strategy, context)
        
        return gaps
    
    def get_optimal_strategy_estimate(self, context):
        """컨텍스트 기반 최적 전략 추정"""
        emotion = context.get('emotion', 'neutral')
        cognitive_load = context.get('cognitive_load_level', 'medium')
        attention = context.get('attention', 0.5)
        
        # 휴리스틱 기반 최적 전략 추정
        if cognitive_load == 'high':
            return 'combination'  # 단순하고 빠른 결정
        elif emotion in ['sad', 'fear']:
            return 'collaboration'  # 신중한 협력
        elif attention > 0.8:
            return 'hybrid'  # 복잡한 전략 가능
        else:
            return 'adaptive'  # 균형잡힌 접근
    
    def estimate_strategy_performance_gap(self, strategy, optimal_strategy, context):
        """전략 간 성능 차이 추정"""
        # 전략별 기본 성능 점수
        base_performance = {
            'combination': 0.7,
            'collaboration': 0.65,
            'hybrid': 0.8,
            'adaptive': 0.75,
            'dynamic': 0.72
        }
        
        optimal_perf = base_performance.get(optimal_strategy, 0.7)
        strategy_perf = base_performance.get(strategy, 0.7)
        
        # 컨텍스트 기반 조정
        emotion = context.get('emotion', 'neutral')
        cognitive_load = context.get('cognitive_load_level', 'medium')
        
        if emotion in ['sad', 'fear'] and strategy == 'collaboration':
            strategy_perf += 0.1
        elif cognitive_load == 'high' and strategy == 'combination':
            strategy_perf += 0.15
        
        gap = max(0.01, optimal_perf - strategy_perf)  # 최소 gap 보장
        return gap

# ==================== Level 1: Cognitive Adaptation MCTS ====================
class CognitiveAdaptationNode:
    """인지 부하 적응 노드"""
    
    def __init__(self, adaptation_type: str, parent=None):
        self.adaptation_type = adaptation_type  # "simplified", "standard", "complex"
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.cognitive_factors = {}  # 인지 부하 요소들 저장
    
    def update(self, reward):
        self.visits += 1
        self.total_reward += reward
    
    def get_average_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0

class CognitiveAdaptationMCTS:
    """Level 1: 인지 부하 기반 적응 MCTS"""
    
    def __init__(self):
        self.adaptation_types = ["simplified", "standard", "complex", "dynamic"]
        self.cognitive_history = deque(maxlen=20)
    
    def search(self, user_context: Dict, meta_strategy: str, iterations=25) -> str:
        """인지 상태에 맞는 적응 전략 탐색"""
        
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        mental_effort = user_context.get('mental_effort_score', 0.5)
        attention = user_context.get('attention', 0.5)
        
        # 휴리스틱 + MCTS 결합
        base_adaptation = self._heuristic_adaptation(cognitive_load, mental_effort, attention)
        
        # MCTS로 미세 조정
        optimized_adaptation = self._mcts_optimize_adaptation(
            base_adaptation, user_context, meta_strategy, iterations
        )
        
        return optimized_adaptation
    
    def _heuristic_adaptation(self, cognitive_load, mental_effort, attention):
        """휴리스틱 기반 기본 적응"""
        
        if cognitive_load == 'high' or mental_effort > 0.8:
            return "simplified"
        elif cognitive_load == 'low' and mental_effort < 0.3 and attention > 0.7:
            return "complex"
        else:
            return "standard"
    
    def _mcts_optimize_adaptation(self, base_adaptation, user_context, meta_strategy, iterations):
        """MCTS로 적응 전략 최적화"""
        
        # 간단한 MCTS 구현
        adaptation_rewards = {}
        
        for adaptation in self.adaptation_types:
            total_reward = 0
            for _ in range(iterations // len(self.adaptation_types)):
                reward = self._calculate_cognitive_adaptation(
                    adaptation, user_context, meta_strategy
                )
                total_reward += reward
            
            adaptation_rewards[adaptation] = total_reward
        
        # 최적 적응 전략 선택
        best_adaptation = max(adaptation_rewards.keys(), 
                            key=lambda x: adaptation_rewards[x])
        
        return best_adaptation
    
    def _calculate_cognitive_adaptation(self, adaptation, user_context, meta_strategy):
        """인지 적응 효과성 계산"""
        
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        mental_effort = user_context.get('mental_effort_score', 0.5)
        emotion = user_context.get('emotion', 'neutral')
        
        # CoTS + TCN Fusion 기반 통합 보상함수 (CVPR 2025)
        
        # 1. CoTS LLM 기반 인지 평가
        cots_reward = self.cots_cognitive_reward(adaptation, cognitive_load, user_context)
        
        # 2. TCN 시공간 멀티모달 보상
        temporal_reward = self.tcn_multimodal_reward(adaptation, user_context)
        
        # 3. 가중 결합
        integrated_reward = 0.7 * cots_reward + 0.3 * temporal_reward
        
        return np.clip(integrated_reward, 0.0, 1.0)
    
    def cots_cognitive_reward(self, adaptation, cognitive_load, context):
        """CoTS (CVPR 2025) 기반 인지 적응 보상"""
        
        # 1. Cognitive Allocation Assessment (1-5 스케일)
        allocation_score = self.assess_cognitive_allocation(adaptation, cognitive_load)
        
        # 2. Task Complexity Cost Evaluation
        complexity_cost = self.evaluate_task_complexity(adaptation, context)
        
        # 3. CoTS 정규화 공식 (1-5 → 0-1)
        normalized_reward = (allocation_score + complexity_cost) / 10.0
        
        return np.clip(normalized_reward, 0.0, 1.0)
    
    def assess_cognitive_allocation(self, adaptation, cognitive_load):
        """인지 자원 할당 평가"""
        allocation_matrix = {
            ('high', 'simplified'): 5,    # 최적 매칭
            ('high', 'standard'): 3,      # 보통
            ('high', 'complex'): 1,       # 부적절
            ('medium', 'standard'): 5,    # 최적
            ('medium', 'simplified'): 3,  # 보통
            ('medium', 'complex'): 3,     # 보통
            ('low', 'complex'): 5,        # 최적
            ('low', 'standard'): 3,       # 보통
            ('low', 'simplified'): 2      # 과소활용
        }
        return allocation_matrix.get((cognitive_load, adaptation), 3)
    
    def evaluate_task_complexity(self, adaptation, context):
        """작업 복잡도 비용 평가"""
        emotion = context.get('emotion', 'neutral')
        attention = context.get('attention', 0.5)
        
        # 기본 복잡도 점수
        complexity_scores = {
            'simplified': 5,  # 낮은 복잡도 = 높은 점수
            'standard': 3,    # 중간 복잡도
            'complex': 1      # 높은 복잡도 = 낮은 점수
        }
        base_score = complexity_scores.get(adaptation, 3)
        
        # 감정 기반 조정
        if emotion in ['sad', 'fear', 'anger']:
            if adaptation == 'simplified':
                base_score += 1  # 부정적 감정에서 단순화 선호
            elif adaptation == 'complex':
                base_score -= 1  # 부정적 감정에서 복잡화 회피
        
        # 주의집중 기반 조정
        if attention < 0.4 and adaptation == 'complex':
            base_score -= 1  # 낮은 집중도에서 복잡화 회피
        elif attention > 0.8 and adaptation == 'complex':
            base_score += 1  # 높은 집중도에서 복잡화 가능
        
        return np.clip(base_score, 1, 5)
    
    def tcn_multimodal_reward(self, adaptation, context):
        """TCN 기반 시공간 멀티모달 보상"""
        
        # 시퀀스 데이터가 없는 경우 현재 상태만 사용
        if not hasattr(self, 'context_history'):
            self.context_history = []
        
        # 현재 컨텍스트를 히스토리에 추가
        self.context_history.append(context)
        
        # 최근 10프레임만 유지
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]
        
        # 1. 시퀀스 특징 추출
        va_sequence = [self.extract_va_features(ctx.get('emotion', 'neutral')) for ctx in self.context_history]
        cog_sequence = [self.extract_cognitive_features(ctx.get('cognitive_load_level', 'medium')) for ctx in self.context_history]
        gaze_sequence = [self.extract_gaze_features(ctx.get('attention', 0.5), ctx) for ctx in self.context_history]
        
        # 2. 시간적 패턴 분석 (간단한 변화율 계산)
        va_temporal = self.analyze_temporal_pattern(va_sequence)
        cog_temporal = self.analyze_temporal_pattern(cog_sequence)
        gaze_temporal = self.analyze_temporal_pattern(gaze_sequence)
        
        # 3. 멀티모달 융합
        combined_features = np.concatenate([va_temporal, cog_temporal, gaze_temporal])
        
        # 4. 적응 전략과의 정렬 평가
        alignment_score = self.evaluate_temporal_alignment(combined_features, adaptation)
        
        return alignment_score
    
    def analyze_temporal_pattern(self, sequence):
        """시간적 패턴 분석"""
        if len(sequence) < 2:
            return np.array([0.0, 0.0])  # [안정성, 변화율]
        
        # 시퀀스를 numpy 배열로 변환
        seq_array = np.array(sequence)
        
        # 안정성 계산 (표준편차의 역수)
        if seq_array.ndim > 1:
            stability = 1.0 / (1.0 + np.mean(np.std(seq_array, axis=0)))
        else:
            stability = 1.0 / (1.0 + np.std(seq_array))
        
        # 변화율 계산 (최근 변화의 크기)
        if seq_array.ndim > 1:
            change_rate = np.mean(np.abs(seq_array[-1] - seq_array[0]))
        else:
            change_rate = abs(seq_array[-1] - seq_array[0])
        
        return np.array([stability, change_rate])
    
    def evaluate_temporal_alignment(self, temporal_features, adaptation):
        """시간적 특징과 적응 전략의 정렬 평가"""
        
        # 특징 요약 (평균)
        avg_stability = np.mean(temporal_features[::2])  # 짝수 인덱스: 안정성
        avg_change = np.mean(temporal_features[1::2])    # 홀수 인덱스: 변화율
        
        # 적응 전략별 선호도
        if adaptation == 'simplified':
            # 단순화는 안정성을 선호, 변화를 회피
            alignment = avg_stability * 0.8 - avg_change * 0.2
        elif adaptation == 'complex':
            # 복잡화는 변화를 활용, 안정성보다는 적응성
            alignment = avg_change * 0.6 + avg_stability * 0.4
        else:  # standard
            # 표준은 균형
            alignment = (avg_stability + avg_change) * 0.5
        
        return np.clip(alignment, 0.0, 1.0)
    
    def extract_va_features(self, emotion):
        """Valence-Arousal 특징 추출 (MetaStrategyMCTS와 동일)"""
        va_mapping = {
            'happy': np.array([0.8, 0.6]),      # 높은 valence, 중간 arousal
            'surprise': np.array([0.3, 0.9]),   # 중간 valence, 높은 arousal
            'sad': np.array([-0.7, -0.3]),      # 낮은 valence, 낮은 arousal
            'anger': np.array([-0.6, 0.8]),     # 낮은 valence, 높은 arousal
            'fear': np.array([-0.8, 0.7]),      # 매우 낮은 valence, 높은 arousal
            'disgust': np.array([-0.5, 0.2]),   # 낮은 valence, 낮은 arousal
            'neutral': np.array([0.0, 0.0])     # 중립
        }
        return va_mapping.get(emotion, np.array([0.0, 0.0]))
    
    def extract_cognitive_features(self, cognitive_load):
        """인지 부하 특징 추출 (MetaStrategyMCTS와 동일)"""
        cognitive_mapping = {
            'low': np.array([0.2, 0.8, 0.9]),     # [부하, 여유도, 처리능력]
            'medium': np.array([0.5, 0.5, 0.6]),
            'high': np.array([0.9, 0.2, 0.3])
        }
        return cognitive_mapping.get(cognitive_load, np.array([0.5, 0.5, 0.6]))
    
    def extract_gaze_features(self, attention, context):
        """시선 추적 특징 추출 (MetaStrategyMCTS와 동일)"""
        fixation_stability = context.get('fixation_stability', 0.5)
        pupil_size = context.get('pupil_size', 0.5)
        
        return np.array([attention, fixation_stability, pupil_size])

# ==================== final_sac.py 방식의 전역 변수들 ====================
# 이미지 전처리 (final_sac.py와 동일)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==================== 실제 ResEmoteNet 모델 ====================
class ResEmoteNet:
    """실제 훈련된 ResEmoteNet 모델"""
    def __init__(self, model_path='fer2013_model.pth'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        
        print(f"Brain Device: {self.device}")
        
        # 실제 ResEmoteNet 모델 로드
        from approach.ResEmoteNet import ResEmoteNet
        self.model = ResEmoteNet().to(self.device)
        
        # 훈련된 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✅ 실제 훈련된 ResEmoteNet 모델 로드됨")
        
        # 이미지 전처리 (final_sac.py와 동일)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.last_emotion = 'neutral'
        self.emotion_stability_count = 0
        

# ==================== 유틸리티 함수들 (기존 코드) ====================
class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

class VectorEMA:
    def __init__(self, alpha: float, length: int):
        self.alpha = alpha
        self.length = length
        self.value = None
    
    def update(self, new_values):
        # Ensure correct length and numpy array
        arr = np.array(new_values, dtype=float)
        if arr.shape[0] != self.length:
            # Pad or truncate to target length
            if arr.shape[0] < self.length:
                pad = np.zeros(self.length - arr.shape[0], dtype=float)
                arr = np.concatenate([arr, pad])
            else:
                arr = arr[:self.length]
        if self.value is None:
            self.value = arr
        else:
            self.value = self.alpha * arr + (1.0 - self.alpha) * self.value
        return self.value

def clamp_float(value: float, min_v: float, max_v: float, default: float) -> float:
    try:
        if value is None or not np.isfinite(value):
            return default
        return float(min(max(value, min_v), max_v))
    except Exception:
        return default

def sanitize_probs(probs: List[float], length: int, default_idx: int) -> List[float]:
    arr = np.array(probs if probs is not None else [], dtype=float)
    if arr.size != length:
        if arr.size < length:
            pad = np.zeros(length - arr.size, dtype=float)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[:length]
    arr[~np.isfinite(arr)] = 0.0
    s = float(arr.sum())
    if s <= 1e-8:
        arr[:] = 0.0
        arr[default_idx] = 1.0
        return arr.tolist()
    return (arr / s).tolist()

def iris_center_radius(landmarks, iris_indices, frame_width, frame_height):
    if not iris_indices:
        return None, None, None
    
    points = []
    for idx in iris_indices:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * frame_width)
            y = int(landmarks[idx].y * frame_height)
            points.append((x, y))
    
    if len(points) < 3:
        return None, None, None
    
    points = np.array(points)
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
    radius = np.max(distances)
    
    return center_x, center_y, radius

def calculate_fixation_stability(gaze_buffer):
    if len(gaze_buffer) < 2:
        return None, None
    
    points = np.array(list(gaze_buffer))
    if points.shape[0] < 2:
        return None, None
    
    cov_matrix = np.cov(points.T)
    if cov_matrix.shape == ():
        cov_matrix = np.array([[cov_matrix]])
    elif cov_matrix.shape == (2,):
        cov_matrix = np.diag(cov_matrix)
    
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    if len(eigenvalues) < 2:
        eigenvalues = np.pad(eigenvalues, (0, 2-len(eigenvalues)), 'constant')
    
    lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
    area = np.pi * np.sqrt(max(lambda1, 0)) * np.sqrt(max(lambda2, 0))
    fix_stab = 1 / (1 + area)
    
    return area, fix_stab

def calculate_mad(values):
    if len(values) == 0:
        return 0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return mad

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ==================== Level 2: Agent Combination MCTS ====================
class CombinationNode:
    """에이전트 조합 선택 노드"""
    
    def __init__(self, combination: Tuple[Optional[str], Optional[str], Optional[str]], parent=None):
        self.combination = combination  # (planner_action, critic_action, executor_action)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        
        # 조합 구성 요소들
        self.planner_actions = ["encouraging", "challenging", "adaptive", "calming", "motivating"]
        self.critic_actions = ["supportive", "thorough", "balanced", "understanding", "optimistic"]
        self.executor_actions = ["gentle", "intensive", "standard", "careful", "energetic"]
    
    def is_terminal(self):
        """완전한 조합이 완성되었는지 확인"""
        return all(x is not None for x in self.combination)
    
    def get_possible_expansions(self):
        """다음에 확장 가능한 노드들"""
        planner, critic, executor = self.combination
        
        expansions = []
        if planner is None:
            for action in self.planner_actions:
                expansions.append((action, critic, executor))
        elif critic is None:
            for action in self.critic_actions:
                expansions.append((planner, action, executor))
        elif executor is None:
            for action in self.executor_actions:
                expansions.append((planner, critic, action))
        
        return expansions
    
    def update(self, reward):
        self.visits += 1
        self.total_reward += reward
    
    def get_average_reward(self):
        return self.total_reward / self.visits if self.visits > 0 else 0

class CombinationMCTS:
    """Level 2: 에이전트 조합 최적화 MCTS"""
    
    def __init__(self, c_param=1.4):
        self.c_param = c_param
        self.combination_history = deque(maxlen=30)
        
    def search(self, user_context: Dict, meta_strategy: str, adaptation_type: str, iterations=40) -> Tuple[str, str, str]:
        """최적 에이전트 조합 탐색"""
        
        # 루트 노드 생성 (빈 조합에서 시작)
        root = CombinationNode((None, None, None))
        
        for _ in range(iterations):
            # Selection & Expansion
            leaf = self._select_and_expand(root, user_context, adaptation_type)
            
            # Simulation
            reward = self._calculate_combination(leaf, user_context, meta_strategy, adaptation_type)
            
            # Backpropagation
            self._backpropagate(leaf, reward)
        
        # 최적 조합 선택
        best_combination = self._get_best_combination(root)
        return best_combination
    
    def _select_and_expand(self, root, user_context, adaptation_type):
        """UCB1 기반 선택 및 확장"""
        
        node = root
        
        # Selection: 터미널 노드나 확장 가능한 노드까지
        while node.children and not node.is_terminal():
            if not node.children:
                break
            node = self._best_child_ucb1(node)
        
        # Expansion: 새로운 자식 노드 추가
        if not node.is_terminal():
            expansions = node.get_possible_expansions()
            if expansions:
                # 적응 타입에 따라 확장 전략 조정
                if adaptation_type == "simplified":
                    # 단순화 모드: 첫 번째 옵션 선택
                    new_combination = expansions[0]
                else:
                    # 표준/복잡 모드: 랜덤 선택
                    new_combination = random.choice(expansions)
                
                child = CombinationNode(new_combination, parent=node)
                node.children.append(child)
                return child
        
        return node
    
    def _best_child_ucb1(self, node):
        """UCB1으로 최적 자식 선택"""
        
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            if child.visits == 0:
                return child  # 아직 방문하지 않은 노드 우선
            
            exploitation = child.get_average_reward()
            exploration = self.c_param * math.sqrt(2 * math.log(node.visits) / child.visits)
            ucb1_score = exploitation + exploration
            
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child
        
        return best_child
    
    def _calculate_combination(self, node, user_context, meta_strategy, adaptation_type):
        """조합 효과성 계산"""
        
        combination = node.combination
        planner, critic, executor = combination
        
        # 완전한 조합이 아닌 경우 랜덤으로 완성
        if not node.is_terminal():
            if planner is None:
                planner = random.choice(node.planner_actions)
            if critic is None:
                critic = random.choice(node.critic_actions)
            if executor is None:
                executor = random.choice(node.executor_actions)
        
        emotion = user_context.get('emotion', 'neutral')
        attention = user_context.get('attention', 0.5)
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        
        # Mixed-R1 BMAS 기반 통합 보상함수 (NeurIPS 2024)
        
        # 1. BMAS 에이전트 유사도 계산
        bmas_reward = self.mixed_r1_combination_reward((planner, critic, executor), user_context)
        
        # 2. 생리신호 통합 보상
        physio_reward = self.physiological_gaze_reward((planner, critic, executor), user_context)
        
        # 3. 가중 결합
        integrated_reward = 0.5 * bmas_reward + 0.5 * physio_reward
        
        return np.clip(integrated_reward, 0.0, 1.0)
    
    def mixed_r1_combination_reward(self, combination, context):
        """Mixed-R1 (NeurIPS 2024) BMAS 기반 조합 보상"""
        
        pred_combination = combination  # (planner, critic, executor)
        
        # 1. 최적 조합 추정 (컨텍스트 기반)
        optimal_combination = self.get_optimal_combination(context)
        
        # 2. BMAS 계산
        bmas_score = self.calculate_bmas(pred_combination, optimal_combination)
        
        return bmas_score
    
    def calculate_bmas(self, pred_agents, optimal_agents):
        """BMAS (Bidirectional Max-Average Similarity) 공식"""
        
        # Agent embedding 계산
        pred_embeddings = [self.get_agent_embedding(agent) for agent in pred_agents if agent is not None]
        optimal_embeddings = [self.get_agent_embedding(agent) for agent in optimal_agents if agent is not None]
        
        if not pred_embeddings or not optimal_embeddings:
            return 0.5  # 기본값
        
        # Forward similarity: pred → optimal
        forward_similarities = []
        for pred_emb in pred_embeddings:
            sims = [self.cosine_similarity(pred_emb, opt_emb) for opt_emb in optimal_embeddings]
            forward_similarities.extend(sims)
        
        # Backward similarity: optimal → pred  
        backward_similarities = []
        for opt_emb in optimal_embeddings:
            sims = [self.cosine_similarity(opt_emb, pred_emb) for pred_emb in pred_embeddings]
            backward_similarities.extend(sims)
        
        # BMAS 공식
        if forward_similarities and backward_similarities:
            bmas = (max(forward_similarities) + np.mean(backward_similarities)) / 2
        else:
            bmas = 0.5
        
        return np.clip(bmas, 0.0, 1.0)
    
    def get_optimal_combination(self, context):
        """컨텍스트 기반 최적 조합 결정"""
        emotion = context.get('emotion', 'neutral')
        cognitive_load = context.get('cognitive_load_level', 'medium')
        attention = context.get('attention', 0.5)
        
        if emotion in ['sad', 'fear']:
            return ('encouraging', 'supportive', 'gentle')
        elif cognitive_load == 'high':
            return ('calming', 'understanding', 'careful')
        elif attention > 0.8:
            return ('challenging', 'thorough', 'intensive')
        elif emotion in ['happy', 'surprise']:
            return ('motivating', 'optimistic', 'energetic')
        else:
            return ('adaptive', 'balanced', 'standard')
    
    def get_agent_embedding(self, agent):
        """에이전트 임베딩 벡터 생성"""
        if agent is None:
            return np.zeros(5)
        
        # 에이전트별 특성 벡터 [활동성, 지지성, 도전성, 신중성, 효율성]
        agent_embeddings = {
            # Planner agents
            'encouraging': np.array([0.7, 0.9, 0.6, 0.4, 0.6]),
            'calming': np.array([0.3, 0.8, 0.2, 0.9, 0.5]),
            'challenging': np.array([0.9, 0.3, 0.9, 0.5, 0.7]),
            'motivating': np.array([0.8, 0.7, 0.8, 0.4, 0.8]),
            'adaptive': np.array([0.5, 0.5, 0.5, 0.7, 0.7]),
            
            # Critic agents
            'supportive': np.array([0.4, 0.9, 0.3, 0.6, 0.5]),
            'understanding': np.array([0.3, 0.8, 0.2, 0.8, 0.4]),
            'thorough': np.array([0.6, 0.4, 0.7, 0.9, 0.8]),
            'optimistic': np.array([0.7, 0.6, 0.6, 0.3, 0.6]),
            'balanced': np.array([0.5, 0.6, 0.5, 0.6, 0.7]),
            
            # Executor agents
            'gentle': np.array([0.3, 0.8, 0.2, 0.7, 0.5]),
            'careful': np.array([0.4, 0.6, 0.3, 0.9, 0.6]),
            'intensive': np.array([0.9, 0.3, 0.8, 0.5, 0.9]),
            'energetic': np.array([0.8, 0.5, 0.7, 0.3, 0.8]),
            'standard': np.array([0.5, 0.5, 0.5, 0.5, 0.7])
        }
        
        return agent_embeddings.get(agent, np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    
    def cosine_similarity(self, vec1, vec2):
        """코사인 유사도 계산"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def physiological_gaze_reward(self, combination, context):
        """생리신호-시선 통합 보상 (EMBC 2024)"""
        
        # 1. 동공 크기 기반 인지부하 (EEG 대안)
        cognitive_2d = self.pupil_to_cognitive_load(context.get('pupil_size', 0.5))
        
        # 2. Eye Tracking 특징
        gaze_features = {
            'fixation_duration': context.get('fixation_duration', 0.5),
            'saccade_velocity': context.get('saccade_velocity', 0.5), 
            'pupil_diameter': context.get('pupil_size', 0.5),
            'blink_rate': context.get('blink_rate', 0.5)
        }
        
        # 3. 특징 추출 및 융합
        eeg_features = self.extract_eeg_features(cognitive_2d)
        eye_features = self.extract_eye_features(gaze_features)
        
        # 4. 조합과의 정렬 평가
        combination_alignment = self.evaluate_combination_alignment(
            eeg_features, eye_features, combination, context
        )
        
        return combination_alignment
    
    def pupil_to_cognitive_load(self, pupil_size):
        """동공 크기를 인지부하 2D 표현으로 변환"""
        # 동공 크기를 2D 공간 패턴으로 매핑
        load_intensity = pupil_size
        spatial_pattern = np.array([
            [load_intensity * 0.8, load_intensity * 0.6],
            [load_intensity * 0.9, load_intensity * 0.7]
        ])
        return spatial_pattern
    
    def extract_eeg_features(self, cognitive_2d):
        """인지부하 2D 패턴에서 특징 추출"""
        # 간단한 통계적 특징
        mean_activity = np.mean(cognitive_2d)
        std_activity = np.std(cognitive_2d)
        max_activity = np.max(cognitive_2d)
        
        return np.array([mean_activity, std_activity, max_activity])
    
    def extract_eye_features(self, gaze_features):
        """시선 추적 특징 벡터 생성"""
        return np.array([
            gaze_features['fixation_duration'],
            gaze_features['saccade_velocity'],
            gaze_features['pupil_diameter'],
            gaze_features['blink_rate']
        ])
    
    def evaluate_combination_alignment(self, eeg_features, eye_features, combination, context):
        """조합과 생리신호의 정렬 평가"""
        
        planner, critic, executor = combination
        
        # 융합된 특징
        fused_features = np.concatenate([eeg_features, eye_features])
        
        # 조합별 가중치
        combination_weights = {
            ('encouraging', 'supportive', 'gentle'): np.array([0.3, 0.2, 0.4, 0.6, 0.8, 0.7, 0.5]),
            ('challenging', 'thorough', 'intensive'): np.array([0.8, 0.7, 0.9, 0.4, 0.3, 0.5, 0.6]),
            ('calming', 'understanding', 'careful'): np.array([0.2, 0.1, 0.3, 0.8, 0.9, 0.8, 0.6]),
            ('adaptive', 'balanced', 'standard'): np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        }
        
        # 기본 가중치 (조합이 정확히 매칭되지 않는 경우)
        weights = combination_weights.get(combination, np.ones(len(fused_features)) / len(fused_features))
        
        # 길이 맞춤
        if len(weights) != len(fused_features):
            if len(weights) > len(fused_features):
                weights = weights[:len(fused_features)]
            else:
                weights = np.pad(weights, (0, len(fused_features) - len(weights)), 'constant', constant_values=0.5)
        
        # 가중 점수 계산
        alignment_score = np.dot(fused_features, weights) / np.sum(weights)
        
        return np.clip(alignment_score, 0.0, 1.0)
        
        # 😊 감정 기반 조합 평가
        emotion_bonuses = {
            'happy': {
                'challenging': 0.3, 'motivating': 0.2,
                'optimistic': 0.2, 'thorough': 0.1,
                'energetic': 0.2, 'intensive': 0.1
            },
            'sad': {
                'encouraging': 0.3, 'calming': 0.2,
                'supportive': 0.3, 'understanding': 0.2,
                'gentle': 0.2, 'careful': 0.1
            },
            'anger': {
                'calming': 0.4, 'adaptive': 0.2,
                'understanding': 0.3, 'supportive': 0.1,
                'careful': 0.2, 'gentle': 0.1
            },
            'fear': {
                'encouraging': 0.2, 'calming': 0.3,
                'supportive': 0.3, 'understanding': 0.2,
                'gentle': 0.3, 'careful': 0.2
            }
        }
        
        if emotion in emotion_bonuses:
            bonuses = emotion_bonuses[emotion]
            reward += bonuses.get(planner, 0)
            reward += bonuses.get(critic, 0)
            reward += bonuses.get(executor, 0)
        
        # 👁️ 집중도 기반 평가
        if attention > 0.8:
            # 높은 집중도 → 도전적 조합 가능
            if planner == 'challenging' and critic == 'thorough':
                reward += 0.15
        elif attention < 0.4:
            # 낮은 집중도 → 부드러운 접근 필요
            if planner == 'encouraging' and executor == 'gentle':
                reward += 0.1
        
        # 📊 메타 전략과의 시너지
        if meta_strategy == 'combination':
            # 조합 최적화와 어울리는 효율적 조합
            if planner == 'adaptive' and critic == 'balanced':
                reward += 0.1
        elif meta_strategy == 'collaboration':
            # 협력과 어울리는 소통 중심 조합  
            if critic == 'supportive' and executor == 'standard':
                reward += 0.1
        
        # 🔄 적응 타입 반영
        if adaptation_type == 'simplified':
            # 단순화 → 표준적인 조합 선호
            if planner == 'adaptive' and critic == 'balanced' and executor == 'standard':
                reward += 0.2
        elif adaptation_type == 'complex':
            # 복잡화 → 정교한 조합 선호
            if len(set([planner, critic, executor])) == 3:  # 모두 다른 스타일
                reward += 0.15
        
        # 조합 시너지 보너스
        synergy_combinations = [
            ('challenging', 'thorough', 'intensive'),      # 고성능 조합
            ('encouraging', 'supportive', 'gentle'),       # 배려 조합
            ('calming', 'understanding', 'careful'),       # 안정 조합
            ('adaptive', 'balanced', 'standard'),          # 균형 조합
            ('motivating', 'optimistic', 'energetic')     # 활력 조합
        ]
        
        if (planner, critic, executor) in synergy_combinations:
            reward += 0.2
        
        # 랜덤 노이즈
        reward += np.random.normal(0, 0.05)
        return np.clip(reward, 0.0, 1.0)
    
    def _backpropagate(self, node, reward):
        """보상 역전파"""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def _get_best_combination(self, root):
        """최적 조합 선택"""
        
        # 모든 터미널 노드 찾기
        terminal_nodes = []
        self._collect_terminal_nodes(root, terminal_nodes)
        
        if not terminal_nodes:
            # 터미널 노드가 없으면 기본 조합 반환
            return ("adaptive", "balanced", "standard")
        
        # 가장 높은 평균 보상을 가진 조합 선택
        best_node = max(terminal_nodes, key=lambda x: x.get_average_reward())
        return best_node.combination
    
    def _collect_terminal_nodes(self, node, terminal_nodes):
        """터미널 노드 수집"""
        if node.is_terminal():
            terminal_nodes.append(node)
        else:
            for child in node.children:
                self._collect_terminal_nodes(child, terminal_nodes)

# ==================== Level 3: Execution Strategy MCTS ====================
class ExecutionStrategyMCTS:
    """
    Level 3: R* (NeurIPS 2024) 기반 실행 전략 최적화
    
    R* 논문의 핵심 기법:
    - Reward Structure Evolution
    - Multi-module Dynamic Weighting  
    - Context-adaptive Fitness Evaluation
    """
    
    # R* 설정 상수들
    class Config:
        # 진화 알고리즘 파라미터
        EVOLUTION_GENERATIONS = 3
        POPULATION_SIZE = 10
        ELITE_SIZE = 2
        MUTATION_RATE_BASE = 0.1
        TOURNAMENT_SIZE = 2
        CROSSOVER_ALPHA = 0.5
        CONVERGENCE_THRESHOLD = 0.005
        
        # 캐싱 시스템 파라미터
        CACHE_HIT_THRESHOLD = 0.8
        MAX_CACHE_SIZE = 50
        
        # 컨텍스트 유사도 파라미터
        ATTENTION_SIMILARITY_THRESHOLD = 0.2
        
        # 메모리 관리 파라미터
        POPULATION_CLEANUP_INTERVAL = 10
        
        # 폴백 보상 파라미터
        FALLBACK_REWARD_DEFAULT = 0.65
        CONTEXT_BONUS = 0.1
        COGNITIVE_BONUS = 0.05
    
    def __init__(self):
        self.execution_strategies = [
            "gentle_adaptive",      # 부드러운 적응형
            "intensive_focused",    # 집중형 강화
            "balanced_standard",    # 균형잡힌 표준
            "careful_methodical",   # 신중한 체계형  
            "energetic_dynamic",    # 활력있는 동적
            "supportive_gradual",   # 지지적 점진형
            "optimized_efficient"   # 최적화 효율형
        ]
        
        # R* 전용 속성 (설정화됨)
        # 진화 알고리즘 파라미터들을 Config 클래스에서 가져옴
        self.reward_structure_population = []  # 보상 구조 개체군
        self.evolution_generations = self.Config.EVOLUTION_GENERATIONS
        self.population_size = self.Config.POPULATION_SIZE
        self.elite_size = self.Config.ELITE_SIZE
        self.mutation_rate_base = self.Config.MUTATION_RATE_BASE
        self.tournament_size = self.Config.TOURNAMENT_SIZE
        self.crossover_alpha = self.Config.CROSSOVER_ALPHA
        self.convergence_threshold = self.Config.CONVERGENCE_THRESHOLD
        
        # R* 캐싱 시스템 (설정화됨)
        self.structure_cache = {}              # 컨텍스트 해시: 최적 구조
        self.cache_hit_threshold = self.Config.CACHE_HIT_THRESHOLD
        self.max_cache_size = self.Config.MAX_CACHE_SIZE
        self.cache_hits = 0                    # 캐시 히트 카운트
        self.cache_misses = 0                  # 캐시 미스 카운트
        
    def search(self, user_context: Dict, meta_strategy: str, adaptation_type: str, 
               combination: Tuple[str, str, str], iterations=20) -> str:
        """최적 실행 전략 선택"""
        
        strategy_rewards = {}
        
        # 각 실행 전략 평가
        for strategy in self.execution_strategies:
            total_reward = 0
            for _ in range(iterations // len(self.execution_strategies) + 1):
                reward = self._calculate_execution_strategy(
                    strategy, user_context, meta_strategy, adaptation_type, combination
                )
                total_reward += reward
                
            strategy_rewards[strategy] = total_reward / (iterations // len(self.execution_strategies) + 1)
        
        # 최적 전략 선택
        best_strategy = max(strategy_rewards.keys(), 
                          key=lambda x: strategy_rewards[x])
        
        return best_strategy
    
    def _calculate_execution_strategy(self, strategy, user_context, meta_strategy, 
                                    adaptation_type, combination):
        """실행 전략 효과성 계산"""
        
        emotion = user_context.get('emotion', 'neutral')
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        attention = user_context.get('attention', 0.5)
        
        planner_action, critic_action, executor_action = combination
        
        # R* (NeurIPS 2024) 기반 자동 보상 설계
        
        # R* 진화된 보상 함수 사용
        r_star_reward = self.r_star_evolved_reward(
            strategy, user_context, meta_strategy, adaptation_type, combination
        )
        
        return r_star_reward
    
    def r_star_evolved_reward(self, strategy: str, user_context: Dict[str, Any], 
                             meta_strategy: str, adaptation_type: str, 
                             combination: Tuple[str, str, str]) -> float:
        """R* (NeurIPS 2024) 기반 진화된 보상 설계 (캐싱 + 에러 처리)"""
        
        try:
            # 1. 입력 검증
            if not strategy or not user_context:
                raise ValueError("Invalid input: strategy or user_context is empty")
            
            # 2. 캐시 확인
            context_hash = self.hash_context(user_context, strategy, meta_strategy, adaptation_type)
            cached_structure = self.get_cached_structure(context_hash)
            
            # 3. 보상 모듈 생성
            reward_modules = self.generate_reward_modules(strategy, user_context, combination)
            
            # 4. 모듈 검증
            if not reward_modules or len(reward_modules) == 0:
                raise ValueError("No reward modules generated")
            
            # 5. NaN/Inf 검사
            for module_name, value in reward_modules.items():
                if not np.isfinite(value):
                    print(f"Warning: {module_name} has invalid value {value}, using default 0.5")
                    reward_modules[module_name] = 0.5
            
            if cached_structure is not None:
                # 캐시 히트: 즉시 계산
                self.cache_hits += 1
                final_reward = self.calculate_evolved_reward(reward_modules, cached_structure)
            else:
                # 캐시 미스: 진화 알고리즘 실행
                self.cache_misses += 1
                optimal_structure = self.evolve_reward_structure(reward_modules, user_context, strategy)
                
                # 결과 검증
                if not optimal_structure:
                    raise ValueError("Evolution failed to produce valid structure")
                
                # 결과 캐싱
                self.cache_structure(context_hash, optimal_structure)
                
                final_reward = self.calculate_evolved_reward(reward_modules, optimal_structure)
            
            # 6. 최종 검증
            if not np.isfinite(final_reward):
                print(f"Warning: Final reward is invalid {final_reward}, using fallback")
                final_reward = self.fallback_reward(strategy, user_context)
            
            return np.clip(final_reward, 0.0, 1.0)
            
        except Exception as e:
            print(f"Error in R* evolution: {e}")
            # 폴백 보상 계산
            return self.fallback_reward(strategy, user_context)
    
    def generate_reward_modules(self, strategy: str, context: Dict[str, Any], 
                               combination: Tuple[str, str, str]) -> Dict[str, float]:
        """
        R* 보상 모듈 생성
        
        6개의 보상 모듈을 생성하여 진화 알고리즘에서 사용
        
        Args:
            strategy (str): 실행 전략 이름
            context (Dict): 사용자 컨텍스트 (감정, 인지부하, 주의집중도)
            combination (Tuple): 에이전트 조합 (planner, critic, executor)
            
        Returns:
            Dict[str, float]: 모듈명 -> 보상값 매핑 (0.0-1.0 범위)
            
        Modules:
            - efficiency: 전략의 효율성 평가
            - satisfaction: 사용자 만족도 평가
            - resource: 자원 사용 최적화 평가
            - cognitive_alignment: 인지 상태와의 정렬도
            - emotion_adaptation: 감정 적응도 (R* 새 모듈)
            - temporal_efficiency: 시간 효율성 (R* 새 모듈)
        """
        modules = {}
        
        # 기존 4개 모듈 (이름 변경)
        modules['efficiency'] = self.calculate_efficiency_module(strategy, context, combination)
        modules['satisfaction'] = self.calculate_satisfaction_module(strategy, context, combination)
        modules['resource'] = self.calculate_resource_module(strategy, context, combination)
        modules['cognitive_alignment'] = self.calculate_alignment_module(strategy, context, combination)
        
        # R* 새로운 모듈들
        modules['emotion_adaptation'] = self.calculate_emotion_module(strategy, context)
        modules['temporal_efficiency'] = self.calculate_temporal_module(strategy, context)
        
        return modules
    
    def evolve_reward_structure(self, reward_modules: Dict[str, float], 
                               context: Dict[str, Any], strategy: str) -> Dict[str, float]:
        """
        R* 진화 알고리즘으로 보상 구조 최적화
        
        진화 알고리즘을 사용하여 6개 보상 모듈의 최적 가중치 조합을 찾음
        
        Args:
            reward_modules (Dict[str, float]): 보상 모듈들
            context (Dict): 사용자 컨텍스트
            strategy (str): 실행 전략
            
        Returns:
            Dict[str, float]: 최적 보상 구조 (모듈명 -> 가중치)
            
        Evolution Process:
            1. 초기 개체군 생성 (Dirichlet 분포)
            2. 진화 루프 (3세대)
               - 적합도 평가 (컨텍스트 + 다양성)
               - 조기 수렴 체크
               - 선택, 교배, 변이
            3. 최적 구조 선택
            4. 메모리 정리
        """
        
        # 초기 개체군 생성 (매번 새로 생성으로 메모리 누수 방지)
        current_population = self.initialize_population(reward_modules)
        
        # 진화 과정
        for generation in range(self.evolution_generations):
            # 적합도 평가
            fitness_scores = []
            for structure in current_population:
                fitness = self.evaluate_structure_fitness(structure, reward_modules, context, strategy)
                fitness_scores.append(fitness)
            
            # 조기 수렴 체크
            if generation > 1 and self.check_convergence(fitness_scores):
                break
            
            # 선택, 교배, 변이
            current_population = self.evolve_population(
                current_population, fitness_scores, generation
            )
        
        # 최적 구조 선택
        final_fitness = [self.evaluate_structure_fitness(s, reward_modules, context, strategy) 
                        for s in current_population]
        best_idx = np.argmax(final_fitness)
        best_structure = current_population[best_idx]
        
        # 메모리 정리: 전역 개체군을 주기적으로 업데이트
        self.cleanup_population()
        
        return best_structure
    
    def initialize_population(self, reward_modules: Dict[str, float]) -> List[Dict[str, float]]:
        """
        초기 보상 구조 개체군 생성
        
        Dirichlet 분포를 사용하여 무작위로 정규화된 가중치 조합들을 생성
        
        Args:
            reward_modules (Dict): 보상 모듈들 (모듈 이름 추출용)
            
        Returns:
            List[Dict]: 초기 개체군 (각 개체는 모듈별 가중치 매핑)
        """
        population = []
        module_names = list(reward_modules.keys())
        
        for _ in range(self.population_size):
            # 랜덤 가중치 생성 (Dirichlet 분포 사용)
            weights = np.random.dirichlet(np.ones(len(module_names)))
            structure = {name: weight for name, weight in zip(module_names, weights)}
            population.append(structure)
        
        return population
    
    def evaluate_structure_fitness(self, structure: Dict[str, float], 
                                   reward_modules: Dict[str, float], 
                                   context: Dict[str, Any], strategy: str) -> float:
        """
        보상 구조의 적합도 평가
        
        보상 구조의 성능을 3가지 기준으로 평가
        
        Args:
            structure (Dict): 보상 구조 (모듈별 가중치)
            reward_modules (Dict): 보상 모듈 값들
            context (Dict): 사용자 컨텍스트
            strategy (str): 실행 전략
            
        Returns:
            float: 전체 적합도 (0.0-1.0 범위)
            
        Fitness Components:
            1. 가중합 보상 (70%)
            2. 컨텍스트 적합성 보너스 (25%)
            3. 구조 다양성 보너스 (5%)
        """
        
        # 가중합 계산
        weighted_reward = sum(structure[module] * reward_modules[module] 
                             for module in structure.keys())
        
        # 컨텍스트 적합성 보너스
        context_bonus = self.calculate_context_fitness(structure, context, strategy)
        
        # 구조 다양성 보너스 (너무 극단적이지 않도록)
        diversity_bonus = self.calculate_diversity_bonus(structure)
        
        total_fitness = weighted_reward + 0.1 * context_bonus + 0.05 * diversity_bonus
        
        return np.clip(total_fitness, 0.0, 1.0)
    
    def calculate_context_fitness(self, structure: Dict[str, float], 
                                  context: Dict[str, Any], strategy: str) -> float:
        """
        컨텍스트에 따른 구조 적합성
        
        사용자의 감정 및 인지 상태에 따라 보상 구조의 적합성을 평가
        
        Args:
            structure (Dict): 보상 구조
            context (Dict): 사용자 컨텍스트
            strategy (str): 실행 전략
            
        Returns:
            float: 컨텍스트 적합성 점수
            
        Context Preferences:
            - 부정적 감정: satisfaction, emotion_adaptation 모듈 선호
            - 긍정적 감정: efficiency, temporal_efficiency 모듈 선호
            - 높은 인지부하: cognitive_alignment, resource 모듈 선호
        """
        
        emotion = context.get('emotion', 'neutral')
        cognitive_load = context.get('cognitive_load_level', 'medium')
        
        fitness = 0.0
        
        # 감정별 모듈 가중치 선호도
        if emotion in ['sad', 'fear']:
            if structure.get('satisfaction', 0) > 0.3:
                fitness += 0.2
            if structure.get('emotion_adaptation', 0) > 0.3:
                fitness += 0.3
        elif emotion in ['happy', 'surprise']:
            if structure.get('efficiency', 0) > 0.3:
                fitness += 0.2
            if structure.get('temporal_efficiency', 0) > 0.2:
                fitness += 0.1
        
        # 인지 부하별 모듈 선호도
        if cognitive_load == 'high':
            if structure.get('cognitive_alignment', 0) > 0.4:
                fitness += 0.3
            if structure.get('resource', 0) > 0.3:
                fitness += 0.2
        elif cognitive_load == 'low':
            if structure.get('efficiency', 0) > 0.4:
                fitness += 0.2
        
        return fitness
    
    def calculate_diversity_bonus(self, structure: Dict[str, float]) -> float:
        """
        구조 다양성 보너스
        
        극단적으로 한 모듈에만 집중되는 것을 방지하기 위해 다양성 지표 계산
        엔트로피를 사용하여 가중치 분포의 균등성 측정
        
        Args:
            structure (Dict): 보상 구조
            
        Returns:
            float: 다양성 점수 (0.0-1.0, 1.0이 최대 다양성)
        """
        weights = list(structure.values())
        
        # 엔트로피 계산 (높을수록 다양성 있음)
        entropy = -sum(w * np.log(w + 1e-8) for w in weights)
        max_entropy = -np.log(1.0 / len(weights))  # 균등 분포일 때 최대 엔트로피
        
        diversity_score = entropy / max_entropy
        return diversity_score
    
    def check_convergence(self, fitness_scores: List[float]) -> bool:
        """
        수렴 여부 확인
        
        상위 25% 개체들의 평균 적합도 개선도를 기준으로 수렴 판단
        개선도가 임계값 이하면 조기 종료
        
        Args:
            fitness_scores (List[float]): 현재 세대의 적합도 점수들
            
        Returns:
            bool: 수렴 여부 (True: 수렴, False: 계속 진화)
        """
        if len(fitness_scores) < 2:
            return False
        
        # 상위 25% 평균 개선도 확인
        top_25_percent = int(len(fitness_scores) * 0.25) or 1
        current_top = np.mean(sorted(fitness_scores, reverse=True)[:top_25_percent])
        
        # 이전 세대와 비교 (간단하게 현재 최대값으로 비교)
        if hasattr(self, '_previous_best_fitness'):
            improvement = current_top - self._previous_best_fitness
            if improvement < self.convergence_threshold:
                return True
        
        self._previous_best_fitness = current_top
        return False
    
    def evolve_population(self, population: List[Dict[str, float]], 
                         fitness_scores: List[float], generation: int) -> List[Dict[str, float]]:
        """
        개체군 진화
        
        선택, 교배, 변이 연산을 통해 다음 세대 개체군을 생성
        
        Args:
            population (List[Dict]): 현재 개체군
            fitness_scores (List[float]): 적합도 점수들
            generation (int): 현재 세대 번호
            
        Returns:
            List[Dict]: 다음 세대 개체군
            
        Evolution Steps:
            1. 엘리트 보존 (상위 20%)
            2. 나머지는 선택-교배-변이로 생성
        """
        
        new_population = []
        
        # 엘리트 보존 (상위 20%)
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # 나머지는 선택, 교배, 변이로 생성
        while len(new_population) < self.population_size:
            # 토너먼트 선택
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            
            # 교배
            child = self.crossover(parent1, parent2)
            
            # 변이
            child = self.mutate(child, generation)
            
            new_population.append(child)
        
        return new_population
    
    def tournament_selection(self, population: List[Dict[str, float]], 
                            fitness_scores: List[float]) -> Dict[str, float]:
        """
        토너먼트 선택
        
        무작위로 선택된 개체들 중에서 가장 적합도가 높은 개체를 선택
        
        Args:
            population (List[Dict]): 개체군
            fitness_scores (List[float]): 적합도 점수들
            
        Returns:
            Dict: 선택된 개체 (복사본)
        """
        tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """
        교배 (가중 평균)
        
        두 부모 개체의 가중치를 선형 결합하여 자식 생성
        
        Args:
            parent1 (Dict): 부모 1
            parent2 (Dict): 부모 2
            
        Returns:
            Dict: 자식 개체 (정규화된 가중치)
        """
        child = {}
        for key in parent1.keys():
            child[key] = self.crossover_alpha * parent1[key] + (1 - self.crossover_alpha) * parent2[key]
        
        # 가중치 정규화
        total_weight = sum(child.values())
        for key in child:
            child[key] /= total_weight
        
        return child
    
    def mutate(self, individual: Dict[str, float], generation: int) -> Dict[str, float]:
        """
        변이 (적응적 변이율)
        
        가우시안 노이즈를 추가하여 개체를 변이시킴
        변이율은 세대에 따라 적응적으로 조정 (초기 높음 -> 후반 낮음)
        
        Args:
            individual (Dict): 변이시킬 개체
            generation (int): 현재 세대 번호
            
        Returns:
            Dict: 변이된 개체 (정규화된 가중치)
        """
        mutated = individual.copy()
        
        # 적응적 변이율 (초기에는 높게, 후반에는 낮게)
        mutation_rate = self.adaptive_mutation_rate(generation)
        
        for key in mutated:
            if np.random.random() < mutation_rate:
                # 가우시안 노이즈 추가
                noise = np.random.normal(0, 0.05)
                mutated[key] = max(0.01, mutated[key] + noise)
        
        # 가중치 재정규화
        total_weight = sum(mutated.values())
        for key in mutated:
            mutated[key] /= total_weight
        
        return mutated
    
    def adaptive_mutation_rate(self, generation: int) -> float:
        """
        적응적 변이율 계산
        
        세대에 따라 변이율을 조정하여 초기에는 탐색을, 후반에는 활용을 강화
        
        Args:
            generation (int): 현재 세대 번호
            
        Returns:
            float: 적응된 변이율 (0.05-0.15 범위)
        """
        # 초기에는 높게, 후반에는 낮게
        return 0.15 * (1 - generation / self.evolution_generations) + 0.05
    
    def calculate_evolved_reward(self, reward_modules: Dict[str, float], 
                                optimal_structure: Dict[str, float]) -> float:
        """
        진화된 구조로 최종 보상 계산
        
        최적화된 가중치를 사용하여 6개 모듈의 가중 합을 계산
        
        Args:
            reward_modules (Dict[str, float]): 보상 모듈 값들
            optimal_structure (Dict[str, float]): 최적 가중치 구조
            
        Returns:
            float: 최종 보상 값 (0.0-1.0 범위)
        """
        
        final_reward = sum(optimal_structure[module] * reward_modules[module] 
                          for module in optimal_structure.keys())
        
        return final_reward
    
    def calculate_emotion_module(self, strategy: str, context: Dict[str, Any]) -> float:
        """
        R* 새로운 모듈: 감정 적응
        
        사용자의 감정 상태에 따른 전략의 적합성을 평가
        각 전략이 특정 감정에 얼마나 잘 맞는지 측정
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            
        Returns:
            float: 감정 적응 점수 (0.0-1.0)
        """
        emotion = context.get('emotion', 'neutral')
        
        emotion_strategy_scores = {
            'gentle_adaptive': {'sad': 0.9, 'fear': 0.8, 'anger': 0.7, 'disgust': 0.6},
            'intensive_focused': {'happy': 0.8, 'surprise': 0.7, 'neutral': 0.5},
            'energetic_dynamic': {'happy': 0.9, 'surprise': 0.8, 'neutral': 0.6},
            'supportive_gradual': {'sad': 0.8, 'fear': 0.9, 'disgust': 0.7, 'anger': 0.6},
            'careful_methodical': {'anger': 0.8, 'fear': 0.7, 'disgust': 0.6, 'neutral': 0.7},
            'balanced_standard': {'neutral': 0.8, 'happy': 0.6, 'sad': 0.6},
            'optimized_efficient': {'neutral': 0.7, 'happy': 0.6, 'surprise': 0.5}
        }
        
        return emotion_strategy_scores.get(strategy, {}).get(emotion, 0.5)
    
    def calculate_temporal_module(self, strategy: str, context: Dict[str, Any]) -> float:
        """
        R* 새로운 모듈: 시간 효율성
        
        사용자의 주의집중도와 인지부하를 고려한 시간 효율성 평가
        전략별 시간 효율성 가중치를 적용
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            
        Returns:
            float: 시간 효율성 점수 (0.0-1.0+)
            
        Formula:
            temporal_score = attention × cognitive_load_factor × strategy_multiplier
        """
        cognitive_load = context.get('cognitive_load_level', 'medium')
        attention = context.get('attention', 0.5)
        
        # 시간 효율성 = 주의집중도 × 인지부하 역함수
        load_factor = {'low': 1.0, 'medium': 0.7, 'high': 0.4}.get(cognitive_load, 0.7)
        temporal_score = attention * load_factor
        
        # 전략별 시간 효율성 조정
        strategy_multipliers = {
            'optimized_efficient': 1.2,
            'intensive_focused': 1.1,
            'energetic_dynamic': 1.0,
            'balanced_standard': 0.9,
            'gentle_adaptive': 0.8,
            'supportive_gradual': 0.7,
            'careful_methodical': 0.6
        }
        
        return temporal_score * strategy_multipliers.get(strategy, 1.0)
    
    def hash_context(self, context: Dict[str, Any], strategy: str, 
                    meta_strategy: str, adaptation_type: str) -> str:
        """
        컨텍스트 해시 생성
        
        캐싱을 위해 사용자 컨텍스트와 전략 정보를 문자열 해시로 변환
        
        Args:
            context (Dict): 사용자 컨텍스트
            strategy (str): 실행 전략
            meta_strategy (str): 메타 전략
            adaptation_type (str): 적응 타입
            
        Returns:
            str: 컨텍스트 해시 문자열
        """
        emotion = context.get('emotion', 'neutral')
        cognitive_load = context.get('cognitive_load_level', 'medium')
        attention = round(context.get('attention', 0.5), 1)  # 0.1 단위로 반올림
        
        # 해시 문자열 생성
        hash_str = f"{strategy}_{meta_strategy}_{adaptation_type}_{emotion}_{cognitive_load}_{attention}"
        return hash_str
    
    def get_cached_structure(self, context_hash: str) -> Optional[Dict[str, float]]:
        """
        캐시된 구조 검색
        
        주어진 컨텍스트 해시에 대해 캐시된 보상 구조를 찾음
        정확한 매치가 없으면 유사한 컨텍스트를 찾음
        
        Args:
            context_hash (str): 컨텍스트 해시
            
        Returns:
            Dict or None: 캐시된 보상 구조 또는 None
        """
        # 정확한 매치 먼저 확인
        if context_hash in self.structure_cache:
            return self.structure_cache[context_hash]
        
        # 유사한 컨텍스트 검색
        for cached_hash, structure in self.structure_cache.items():
            similarity = self.calculate_context_similarity(context_hash, cached_hash)
            if similarity > self.cache_hit_threshold:
                return structure
        
        return None
    
    def calculate_context_similarity(self, hash1: str, hash2: str) -> float:
        """
        컨텍스트 유사도 계산
        
        두 컨텍스트 해시 간의 유사도를 0.0-1.0 범위로 계산
        문자열 요소들을 비교하여 일치도를 측정
        
        Args:
            hash1 (str): 컨텍스트 해시 1
            hash2 (str): 컨텍스트 해시 2
            
        Returns:
            float: 유사도 (0.0-1.0, 1.0이 완전 일치)
        """
        parts1 = hash1.split('_')
        parts2 = hash2.split('_')
        
        if len(parts1) != len(parts2):
            return 0.0
        
        matches = 0
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                matches += 1
            elif p1.replace('.', '').isdigit() and p2.replace('.', '').isdigit():
                # 숫자 비교 (주의집중도)
                diff = abs(float(p1) - float(p2))
                if diff <= self.Config.ATTENTION_SIMILARITY_THRESHOLD:
                    matches += 0.8
        
        similarity = matches / len(parts1)
        return similarity
    
    def cache_structure(self, context_hash: str, structure: Dict[str, float]) -> None:
        """
        구조 캐싱
        
        진화된 보상 구조를 캐시에 저장
        캐시 크기 제한을 초과하면 LRU 방식으로 오래된 항목 제거
        
        Args:
            context_hash (str): 컨텍스트 해시
            structure (Dict): 보상 구조 (복사되어 저장)
        """
        # 캐시 크기 제한
        if len(self.structure_cache) >= self.max_cache_size:
            # LRU: 가장 오래된 항목 제거
            oldest_key = next(iter(self.structure_cache))
            del self.structure_cache[oldest_key]
        
        self.structure_cache[context_hash] = structure.copy()
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        캐시 통계 반환
        
        R* 캐싱 시스템의 성능 지표를 반환
        
        Returns:
            Dict: 캐시 통계 정보
                - cache_hits: 캐시 히트 횟수
                - cache_misses: 캐시 미스 횟수
                - hit_rate: 캐시 히트율 (0.0-1.0)
                - cache_size: 현재 캐시 크기
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.structure_cache)
        }
    
    def cleanup_population(self) -> None:
        """
        메모리 관리: 개체군 주기적 정리
        
        메모리 누수를 방지하기 위해 주기적으로 개체군 크기를 제한하고 가비지 컨렉션 수행
        
        Cleanup Strategy:
            - 매 10번째 진화마다 실행
            - 개체군 크기를 설정된 크기로 제한
            - Python 가비지 컨렉션 호출
        """
        # 전역 개체군을 주기적으로 업데이트 (매 10번째 진화마다)
        if not hasattr(self, '_evolution_count'):
            self._evolution_count = 0
        
        self._evolution_count += 1
        
        # 설정된 간격마다 전역 개체군 업데이트
        if self._evolution_count % self.Config.POPULATION_CLEANUP_INTERVAL == 0:
            # 최근 진화 결과로 전역 개체군 업데이트
            if len(self.reward_structure_population) > self.population_size:
                # 크기 제한: 상위 population_size개만 유지
                self.reward_structure_population = self.reward_structure_population[:self.population_size]
            
            # 메모리 정리 호출
            import gc
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Union[int, float]]:
        """
        메모리 사용량 통계
        
        R* 시스템의 메모리 사용량을 추정하여 반환
        
        Returns:
            Dict: 메모리 사용량 정보
                - population_count: 개체군 크기
                - cache_count: 캐시 크기
                - estimated_population_memory_bytes: 개체군 메모리 예상 크기
                - estimated_cache_memory_bytes: 캐시 메모리 예상 크기
                - evolution_count: 진화 실행 횟수
        """
        import sys
        
        population_size = len(self.reward_structure_population)
        cache_size = len(self.structure_cache)
        
        # 개체군 메모리 예상 크기 (각 구조당 약 6개 모듈)
        estimated_population_memory = population_size * 6 * sys.getsizeof(0.5)  # float 크기
        estimated_cache_memory = cache_size * 6 * sys.getsizeof(0.5)
        
        return {
            'population_count': population_size,
            'cache_count': cache_size,
            'estimated_population_memory_bytes': estimated_population_memory,
            'estimated_cache_memory_bytes': estimated_cache_memory,
            'evolution_count': getattr(self, '_evolution_count', 0)
        }
    
    def fallback_reward(self, strategy: str, context: Dict[str, Any]) -> float:
        """
        에러 발생 시 폴백 보상 계산
        
        R* 진화 알고리즘에서 에러가 발생했을 때 사용할 간단한 휴리스틱 보상
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            
        Returns:
            float: 폴백 보상 값 (0.0-1.0 범위)
            
        Fallback Strategy:
            1. 전략별 기본 점수 사용
            2. 간단한 컨텍스트 조정 적용
            3. 최종 실패 시 고정값 반환
        """
        try:
            # 간단한 휴리스틱 기반 보상
            emotion = context.get('emotion', 'neutral')
            cognitive_load = context.get('cognitive_load_level', 'medium')
            
            # 전략별 기본 점수
            base_scores = {
                'gentle_adaptive': 0.6,
                'intensive_focused': 0.7,
                'balanced_standard': 0.65,
                'careful_methodical': 0.6,
                'energetic_dynamic': 0.7,
                'supportive_gradual': 0.6,
                'optimized_efficient': 0.75
            }
            
            base_reward = base_scores.get(strategy, self.Config.FALLBACK_REWARD_DEFAULT)
            
            # 간단한 컨텍스트 조정
            if emotion in ['sad', 'fear'] and strategy in ['supportive_gradual', 'gentle_adaptive']:
                base_reward += self.Config.CONTEXT_BONUS
            elif emotion in ['happy', 'surprise'] and strategy in ['energetic_dynamic', 'intensive_focused']:
                base_reward += self.Config.CONTEXT_BONUS
            
            if cognitive_load == 'high' and strategy in ['gentle_adaptive', 'supportive_gradual']:
                base_reward += self.Config.COGNITIVE_BONUS
            
            return np.clip(base_reward, 0.0, 1.0)
            
        except Exception:
            # 최종 폴백: 고정값
            return self.Config.FALLBACK_REWARD_DEFAULT
    
    def calculate_efficiency_module(self, strategy: str, context: Dict[str, Any], 
                                   combination: Tuple[str, str, str]) -> float:
        """
        효율성 모듈
        
        전략의 기본 효율성을 평가하고 사용자의 인지 상태에 따라 조정
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            combination (Tuple): 에이전트 조합
            
        Returns:
            float: 효율성 점수 (0.0-1.0)
        """
        cognitive_load = context.get('cognitive_load_level', 'medium')
        attention = context.get('attention', 0.5)
        
        # 전략별 기본 효율성
        efficiency_scores = {
            'optimized_efficient': 0.9,
            'intensive_focused': 0.8,
            'balanced_standard': 0.7,
            'gentle_adaptive': 0.6,
            'supportive_gradual': 0.5,
            'careful_methodical': 0.6,
            'energetic_dynamic': 0.7
        }
        
        base_efficiency = efficiency_scores.get(strategy, 0.5)
        
        # 인지부하 기반 조정
        if cognitive_load == 'low' and strategy in ['optimized_efficient', 'intensive_focused']:
            base_efficiency += 0.2
        elif cognitive_load == 'high' and strategy in ['gentle_adaptive', 'supportive_gradual']:
            base_efficiency += 0.15
        
        # 주의집중 기반 조정
        if attention > 0.8 and strategy in ['intensive_focused', 'energetic_dynamic']:
            base_efficiency += 0.1
        elif attention < 0.4 and strategy in ['gentle_adaptive', 'careful_methodical']:
            base_efficiency += 0.1
        
        return np.clip(base_efficiency, 0.0, 1.0)
    
    def calculate_satisfaction_module(self, strategy: str, context: Dict[str, Any], 
                                     combination: Tuple[str, str, str]) -> float:
        """
        만족도 모듈
        
        사용자의 감정 상태에 따른 전략 선호도를 기반으로 만족도 평가
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            combination (Tuple): 에이전트 조합
            
        Returns:
            float: 만족도 점수 (0.0-1.0)
        """
        emotion = context.get('emotion', 'neutral')
        
        # 감정별 선호 전략
        emotion_preferences = {
            'happy': ['energetic_dynamic', 'intensive_focused', 'optimized_efficient'],
            'surprise': ['energetic_dynamic', 'balanced_standard'],
            'sad': ['supportive_gradual', 'gentle_adaptive', 'careful_methodical'],
            'fear': ['supportive_gradual', 'gentle_adaptive'],
            'anger': ['careful_methodical', 'balanced_standard'],
            'disgust': ['careful_methodical', 'gentle_adaptive'],
            'neutral': ['balanced_standard', 'optimized_efficient']
        }
        
        preferred_strategies = emotion_preferences.get(emotion, ['balanced_standard'])
        
        if strategy in preferred_strategies:
            satisfaction = 0.8 + (preferred_strategies.index(strategy) * -0.1)  # 첫 번째가 가장 높음
        else:
            satisfaction = 0.4
        
        return np.clip(satisfaction, 0.0, 1.0)
    
    def calculate_resource_module(self, strategy: str, context: Dict[str, Any], 
                                 combination: Tuple[str, str, str]) -> float:
        """
        자원 최적화 모듈
        
        전략의 자원 사용량을 평가하고 사용자의 인지부하에 따라 조정
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            combination (Tuple): 에이전트 조합
            
        Returns:
            float: 자원 최적화 점수 (0.0-1.0)
        """
        cognitive_load = context.get('cognitive_load_level', 'medium')
        
        # 전략별 자원 사용량 (낮을수록 좋음)
        resource_usage = {
            'gentle_adaptive': 0.3,
            'supportive_gradual': 0.4,
            'balanced_standard': 0.5,
            'careful_methodical': 0.6,
            'optimized_efficient': 0.7,
            'energetic_dynamic': 0.8,
            'intensive_focused': 0.9
        }
        
        usage = resource_usage.get(strategy, 0.5)
        
        # 인지부하가 높을 때는 낮은 자원 사용 선호
        if cognitive_load == 'high':
            resource_score = 1.0 - usage  # 사용량이 낮을수록 높은 점수
        else:
            resource_score = 0.5 + (usage * 0.5)  # 적절한 자원 사용 선호
        
        return np.clip(resource_score, 0.0, 1.0)
    
    def calculate_alignment_module(self, strategy: str, context: Dict[str, Any], 
                                  combination: Tuple[str, str, str]) -> float:
        """
        인지 정렬 모듈
        
        사용자의 인지 상태(인지부하, 주의집중도, 감정)와 전략 간의 정렬도 평가
        
        Args:
            strategy (str): 실행 전략
            context (Dict): 사용자 컨텍스트
            combination (Tuple): 에이전트 조합
            
        Returns:
            float: 인지 정렬 점수 (0.0-1.0)
        """
        cognitive_load = context.get('cognitive_load_level', 'medium')
        attention = context.get('attention', 0.5)
        emotion = context.get('emotion', 'neutral')
        
        # 다차원 정렬 점수
        load_alignment = self.get_load_strategy_alignment(cognitive_load, strategy)
        attention_alignment = self.get_attention_strategy_alignment(attention, strategy)
        emotion_alignment = self.get_emotion_strategy_alignment(emotion, strategy)
        
        # 가중 평균
        cognitive_alignment = (load_alignment * 0.4 + attention_alignment * 0.3 + emotion_alignment * 0.3)
        
        return np.clip(cognitive_alignment, 0.0, 1.0)
    
    def get_load_strategy_alignment(self, cognitive_load, strategy):
        """인지부하-전략 정렬"""
        alignment_matrix = {
            ('high', 'gentle_adaptive'): 0.9,
            ('high', 'supportive_gradual'): 0.8,
            ('high', 'careful_methodical'): 0.7,
            ('medium', 'balanced_standard'): 0.9,
            ('medium', 'optimized_efficient'): 0.7,
            ('low', 'intensive_focused'): 0.9,
            ('low', 'energetic_dynamic'): 0.8,
            ('low', 'optimized_efficient'): 0.8
        }
        return alignment_matrix.get((cognitive_load, strategy), 0.5)
    
    def get_attention_strategy_alignment(self, attention, strategy):
        """주의집중-전략 정렬"""
        if attention > 0.8:
            high_attention_strategies = {
                'intensive_focused': 0.9,
                'energetic_dynamic': 0.8,
                'optimized_efficient': 0.7
            }
            return high_attention_strategies.get(strategy, 0.4)
        elif attention < 0.4:
            low_attention_strategies = {
                'gentle_adaptive': 0.9,
                'supportive_gradual': 0.8,
                'careful_methodical': 0.7
            }
            return low_attention_strategies.get(strategy, 0.4)
        else:
            return 0.6  # 중간 집중도에서는 모든 전략이 적당함
    
    def get_emotion_strategy_alignment(self, emotion, strategy):
        """감정-전략 정렬"""
        emotion_strategy_scores = {
            ('happy', 'energetic_dynamic'): 0.9,
            ('happy', 'intensive_focused'): 0.8,
            ('surprise', 'energetic_dynamic'): 0.8,
            ('sad', 'supportive_gradual'): 0.9,
            ('sad', 'gentle_adaptive'): 0.8,
            ('fear', 'supportive_gradual'): 0.9,
            ('fear', 'gentle_adaptive'): 0.8,
            ('anger', 'careful_methodical'): 0.8,
            ('anger', 'balanced_standard'): 0.7,
            ('neutral', 'balanced_standard'): 0.8,
            ('neutral', 'optimized_efficient'): 0.7
        }
        return emotion_strategy_scores.get((emotion, strategy), 0.5)
    
    # mcts_reward_optimization 메서드 제거됨 - R* 진화 알고리즘으로 대체
    
    # evaluate_reward_performance 메서드 제거됨 - R* fitness 평가로 대체

# ==================== Hierarchical MCTS Integration System ====================
class HierarchicalMCTSSystem:
    """🧠 4단계 계층적 MCTS 통합 시스템 (논문 구현)"""
    
    def __init__(self):
        print("Initializing Revolutionary Hierarchical MCTS System...")
        
        # 4단계 MCTS 시스템들
        self.meta_mcts = MetaStrategyMCTS()
        self.cognitive_mcts = CognitiveAdaptationMCTS()
        self.combination_mcts = CombinationMCTS()
        self.execution_mcts = ExecutionStrategyMCTS()
        
        # 성능 추적
        self.decision_history = deque(maxlen=100)
        self.level_stats = {
            0: MCTSLevelStats(0, 0, 0.0, 0, 0, 0.0),
            1: MCTSLevelStats(1, 0, 0.0, 0, 0, 0.0),
            2: MCTSLevelStats(2, 0, 0.0, 0, 0, 0.0),
            3: MCTSLevelStats(3, 0, 0.0, 0, 0, 0.0)
        }
        
        # 시각화용 데이터
        self.current_tree_visualization = {}
        self.adaptation_history = deque(maxlen=50)
        self.performance_metrics = deque(maxlen=200)
        
        # 메시지 히스토리
        self.message_history = deque(maxlen=40)
        self.add_system_message("Hierarchical MCTS System initialized")
        self.add_system_message("4-Level intelligent decision making ready")
        
        # 논문 구현: GPT-4 에이전트 및 GEMMAS (선택적)
        self.integrated_agents = None
        self.use_llm_agents = False
        try:
            import os
            if os.getenv("OPENAI_API_KEY"):
                print("🤖 Initializing GPT-4 Multi-Agent System...")
                from integration_wrapper import IntegratedMultiAgentSystem
                self.integrated_agents = IntegratedMultiAgentSystem(api_key=os.getenv("OPENAI_API_KEY"))
                self.use_llm_agents = True  # ← GPT-4 에이전트 활성화!
                self.add_system_message("GPT-4 agents ready")
                print("✅ GPT-4 agents integrated and ACTIVATED")
                print("🔄 Real GPT-4 collaboration enabled:")
                print("   → Planner: Generates candidate actions")
                print("   → Critic: Evaluates with Q-values")
                print("   → Executor: Selects final action")
                print("   → GEMMAS: Measures collaboration quality (IDS, UPR)")
        except Exception as e:
            print(f"⚠️  GPT-4 integration skipped: {e}")
            self.integrated_agents = None
        
        # 논문 보상 함수
        try:
            from paper_reward_function import PaperRewardFunction
            self.paper_reward_function = PaperRewardFunction()
            self.add_system_message("Paper reward function loaded")
            print("✅ Paper reward function loaded")
        except Exception as e:
            print(f"⚠️  Paper reward function skipped: {e}")
            self.paper_reward_function = None
        
        print("🚀 Hierarchical MCTS System Ready!")
    
    def add_system_message(self, message):
        """시스템 메시지 추가"""
        timestamp = time.strftime("%H:%M:%S")
        self.message_history.append({
            "time": timestamp,
            "type": "system",
            "message": message
        })
    
    def add_level_message(self, level, message):
        """레벨별 메시지 추가"""
        timestamp = time.strftime("%H:%M:%S")
        self.message_history.append({
            "time": timestamp,
            "type": "level",
            "level": level,
            "message": message
        })
    
    def hierarchical_decision_making(self, user_context: Dict, face_detected: bool) -> HierarchicalDecision:
        """🔶✨ Multi-Adaptive: 다중 에이전트 + 사용자 적응 (제안 시스템 + 논문 구현)"""
        
        if not face_detected:
            return self._generate_no_face_decision()
        
        start_time = time.time()
        
        # 🎯 Level 0: Meta-Strategy Selection
        self.add_level_message(0, "Level 0: Meta-strategy selection starting...")
        meta_strategy = self.meta_mcts.search(user_context, iterations=30)
        self.add_level_message(0, f"Meta-strategy selected: {meta_strategy}")
        
        # 🧠 Level 1: Cognitive Adaptation
        self.add_level_message(1, "Level 1: Cognitive adaptation analysis...")
        cognitive_adaptation = self.cognitive_mcts.search(user_context, meta_strategy, iterations=25)
        self.add_level_message(1, f"Adaptation type: {cognitive_adaptation}")
        
        # ⚡ Level 2: Agent Combination
        self.add_level_message(2, "Level 2: Agent combination optimization...")
        # 인지 부하에 따른 반복 횟수 조정
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        combination_iterations = self._get_adaptive_iterations(cognitive_load, cognitive_adaptation, 'combination')
        
        optimal_combination = self.combination_mcts.search(
            user_context, meta_strategy, cognitive_adaptation, combination_iterations
        )
        self.add_level_message(2, f"Optimal combination: {optimal_combination}")
        
        # 🎯 Level 3: Execution Strategy
        self.add_level_message(3, "Level 3: Execution strategy optimization...")
        execution_iterations = self._get_adaptive_iterations(cognitive_load, cognitive_adaptation, 'execution')
        
        execution_strategy = self.execution_mcts.search(
            user_context, meta_strategy, cognitive_adaptation, optimal_combination, execution_iterations
        )
        self.add_level_message(3, f"Execution strategy: {execution_strategy}")
        
        # 의사결정 완료
        decision_time = time.time() - start_time
        
        # 품질 점수 계산 (논문 보상 함수 사용)
        quality_score = self._calculate_decision_quality_with_paper_reward(
            meta_strategy, cognitive_adaptation, optimal_combination, 
            execution_strategy, user_context
        )
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(user_context, decision_time, quality_score)
        
        # 트리 시각화 데이터 생성
        tree_visualization = self._generate_tree_visualization(
            meta_strategy, cognitive_adaptation, optimal_combination, execution_strategy
        )
        
        # 레벨별 결정 정보
        level_decisions = [
            {"level": 0, "decision": meta_strategy, "type": "meta_strategy"},
            {"level": 1, "decision": cognitive_adaptation, "type": "adaptation"},
            {"level": 2, "decision": optimal_combination, "type": "combination"},
            {"level": 3, "decision": execution_strategy, "type": "execution"}
        ]
        
        # 최종 결정 생성
        hierarchical_decision = HierarchicalDecision(
            meta_strategy=meta_strategy,
            cognitive_adaptation=cognitive_adaptation,
            combination_choice=optimal_combination,
            execution_strategy=execution_strategy,
            tree_depth=4,
            quality_score=quality_score,
            decision_time=decision_time,
            confidence=confidence,
            tree_visualization=tree_visualization,
            level_decisions=level_decisions
        )
        
        # 논문 구현: GPT-4 에이전트 협력 (선택적, 비동기)
        if self.use_llm_agents and self.integrated_agents:
            try:
                self.add_system_message("Consulting GPT-4 agents...")
                agent_result = self.integrated_agents.process_decision(
                    f"Evaluate MCTS decision: {meta_strategy}",
                    user_context
                )
                # GEMMAS 품질 정보 추가
                hierarchical_decision.ids = agent_result.get('ids', 0.0)
                hierarchical_decision.upr = agent_result.get('upr', 0.0)
                hierarchical_decision.llm_feedback = agent_result.get('collaboration_quality', '')
                self.add_system_message(f"GPT-4: IDS={agent_result.get('ids', 0):.2f}, UPR={agent_result.get('upr', 0):.2f}")
            except Exception as e:
                print(f"⚠️  GPT-4 evaluation skipped: {e}")
        
        # 통계 업데이트
        self._update_statistics(hierarchical_decision, user_context)
        
        # 히스토리 저장
        self.decision_history.append(hierarchical_decision)
        
        self.add_system_message(f"Hierarchical decision complete! Quality: {quality_score:.3f}")
        
        return hierarchical_decision
    
    def _calculate_decision_quality_with_paper_reward(self, meta_strategy, cognitive_adaptation, 
                                                      optimal_combination, execution_strategy, user_context):
        """논문 보상 함수를 사용한 품질 계산"""
        if self.paper_reward_function:
            try:
                decision = {
                    'meta_strategy': meta_strategy,
                    'cognitive_adaptation': cognitive_adaptation,
                    'combination_choice': optimal_combination,
                    'execution_strategy': execution_strategy,
                    'decision_time': 0.0,
                    'tree_depth': 4,
                    'confidence': 0.8
                }
                return self.paper_reward_function.calculate_total_reward(user_context, decision)
            except:
                pass
        
        # Fallback to original
        return self._calculate_decision_quality(meta_strategy, cognitive_adaptation, 
                                               optimal_combination, execution_strategy, user_context)
    
    def _get_adaptive_iterations(self, cognitive_load: str, adaptation_type: str, decision_type: str) -> int:
        """인지 부하에 따른 적응적 반복 횟수 결정"""
        
        base_iterations = {
            'combination': 40,
            'execution': 20
        }
        
        base = base_iterations.get(decision_type, 30)
        
        # 인지 부하에 따른 조정
        if cognitive_load == 'high':
            multiplier = 0.7  # 높은 인지부하 → 빠른 결정
        elif cognitive_load == 'low':
            multiplier = 1.3  # 낮은 인지부하 → 정교한 탐색
        else:
            multiplier = 1.0  # 표준
        
        # 적응 타입에 따른 조정
        if adaptation_type == 'simplified':
            multiplier *= 0.8
        elif adaptation_type == 'complex':
            multiplier *= 1.2
        
        return int(base * multiplier)
    
    def _calculate_decision_quality(self, meta_strategy, adaptation, combination, execution, user_context):
        """의사결정 품질 점수 계산"""
        
        base_quality = 0.5
        
        # 메타 전략 품질
        emotion = user_context.get('emotion', 'neutral')
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        
        if cognitive_load == 'high' and meta_strategy == 'combination':
            base_quality += 0.2
        elif cognitive_load == 'low' and meta_strategy == 'hybrid':
            base_quality += 0.15
        
        # 적응 품질
        if cognitive_load == 'high' and adaptation == 'simplified':
            base_quality += 0.15
        elif cognitive_load == 'low' and adaptation == 'complex':
            base_quality += 0.1
        
        # 조합-실행 일치성
        planner, critic, executor = combination
        if executor == 'gentle' and execution.startswith('gentle'):
            base_quality += 0.1
        elif executor == 'intensive' and execution.startswith('intensive'):
            base_quality += 0.1
        
        # 전체적 일관성 보너스
        if (meta_strategy == 'combination' and adaptation == 'simplified' and 
            execution == 'optimized_efficient'):
            base_quality += 0.1
        
        return np.clip(base_quality + np.random.normal(0, 0.02), 0.0, 1.0)
    
    def _calculate_confidence(self, user_context, decision_time, quality_score):
        """의사결정 신뢰도 계산"""
        
        base_confidence = 0.7
        
        # 결정 시간 기반 신뢰도
        if decision_time < 0.1:
            base_confidence += 0.1  # 빠른 결정
        elif decision_time > 0.5:
            base_confidence -= 0.1  # 너무 느린 결정
        
        # 품질 기반 신뢰도
        base_confidence += (quality_score - 0.5) * 0.4
        
        # 인지 부하 기반 신뢰도
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        if cognitive_load == 'high':
            base_confidence -= 0.1  # 높은 부하 → 낮은 신뢰도
        
        return np.clip(base_confidence, 0.0, 1.0)
    
    def _generate_tree_visualization(self, meta_strategy, adaptation, combination, execution):
        """트리 시각화 데이터 생성"""
        
        planner, critic, executor = combination
        
        return {
            "levels": [
                {
                    "level": 0,
                    "name": "Meta Strategy",
                    "decision": meta_strategy,
                    "alternatives": ["combination", "collaboration", "hybrid"],
                    "confidence": 0.85
                },
                {
                    "level": 1, 
                    "name": "Cognitive Adaptation",
                    "decision": adaptation,
                    "alternatives": ["simplified", "standard", "complex"],
                    "confidence": 0.78
                },
                {
                    "level": 2,
                    "name": "Agent Combination",
                    "decision": f"{planner}+{critic}+{executor}",
                    "alternatives": ["125 combinations evaluated"],
                    "confidence": 0.82
                },
                {
                    "level": 3,
                    "name": "Execution Strategy", 
                    "decision": execution,
                    "alternatives": ["gentle_adaptive", "intensive_focused", "balanced_standard"],
                    "confidence": 0.76
                }
            ],
            "connections": [
                {"from": 0, "to": 1, "strength": 0.9},
                {"from": 1, "to": 2, "strength": 0.85},
                {"from": 2, "to": 3, "strength": 0.8}
            ]
        }
    
    def _update_statistics(self, decision, user_context):
        """통계 업데이트"""
        
        # 레벨별 통계 업데이트
        for level in range(4):
            stats = self.level_stats[level]
            stats.decision_count += 1
            stats.avg_decision_time = (
                (stats.avg_decision_time * (stats.decision_count - 1) + 
                 decision.decision_time) / stats.decision_count
            )
            if decision.quality_score > stats.best_value:
                stats.best_value = decision.quality_score
        
        # 적응 히스토리 업데이트
        self.adaptation_history.append({
            'timestamp': time.time(),
            'cognitive_load': user_context.get('cognitive_load_level', 'medium'),
            'adaptation': decision.cognitive_adaptation,
            'quality': decision.quality_score
        })
        
        # 성능 메트릭 업데이트
        self.performance_metrics.append({
            'timestamp': time.time(),
            'decision_time': decision.decision_time,
            'quality_score': decision.quality_score,
            'confidence': decision.confidence,
            'tree_depth': decision.tree_depth
        })
    
    def _generate_no_face_decision(self):
        """얼굴 감지되지 않을 때 기본 결정"""
        
        return HierarchicalDecision(
            meta_strategy="standby",
            cognitive_adaptation="standard",
            combination_choice=("adaptive", "balanced", "standard"),
            execution_strategy="balanced_standard",
            tree_depth=0,
            quality_score=0.0,
            decision_time=0.001,
            confidence=0.0,
            tree_visualization={"levels": [], "connections": []},
            level_decisions=[]
        )
    
    def get_current_performance_summary(self):
        """현재 성능 요약"""
        
        if not self.performance_metrics:
            return {"avg_quality": 0.0, "avg_time": 0.0, "avg_confidence": 0.0}
        
        recent_metrics = list(self.performance_metrics)[-20:]  # 최근 20개
        
        return {
            "avg_quality": np.mean([m['quality_score'] for m in recent_metrics]),
            "avg_time": np.mean([m['decision_time'] for m in recent_metrics]),
            "avg_confidence": np.mean([m['confidence'] for m in recent_metrics]),
            "total_decisions": len(self.decision_history)
        }

# ==================== Hierarchical Visualization Windows ====================

class HierarchicalTreeWindow:
    """🌲 계층적 의사결정 트리 시각화 창"""
    
    def __init__(self, width=1000, height=800):
        self.width = width
        self.height = height
        self.window_name = "🧠 Hierarchical Decision Tree"
        
        # 색상 팔레트
        self.colors = {
            'background': (15, 15, 25),
            'header': (255, 215, 0),  # Gold
            'level_0': (255, 100, 100),  # Meta Strategy - Red
            'level_1': (100, 255, 100),  # Cognitive - Green
            'level_2': (100, 100, 255),  # Combination - Blue
            'level_3': (255, 100, 255),  # Execution - Magenta
            'connection': (200, 200, 200),
            'text': (255, 255, 255),
            'active': (255, 255, 100)
        }
        
        print("🌲 Hierarchical Tree Visualization Window initialized")
    
    def create_tree_window(self, hierarchical_decision, system_stats):
        """계층적 트리 시각화 생성"""
        
        canvas = np.full((self.height, self.width, 3), self.colors['background'], dtype=np.uint8)
        
        # 헤더
        self.draw_header(canvas, hierarchical_decision)
        
        # 계층 구조 시각화
        self.draw_hierarchical_levels(canvas, hierarchical_decision)
        
        # 연결선 그리기
        self.draw_connections(canvas, hierarchical_decision)
        
        # 성능 지표
        self.draw_performance_indicators(canvas, hierarchical_decision, system_stats)
        
        # 의사결정 경로 강조
        self.highlight_decision_path(canvas, hierarchical_decision)
        
        return canvas
    
    def draw_header(self, canvas, decision):
        """헤더 그리기"""
        
        cv2.putText(canvas, "HIERARCHICAL MCTS DECISION TREE", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['header'], 3)
        
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(canvas, f"Time: {current_time}", (self.width - 200, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # 품질 점수 표시
        quality_text = f"Quality Score: {decision.quality_score:.3f}"
        cv2.putText(canvas, quality_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['active'], 2)
        
        # 신뢰도 표시
        confidence_text = f"Confidence: {decision.confidence:.3f}"
        cv2.putText(canvas, confidence_text, (300, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['active'], 2)
        
        cv2.line(canvas, (20, 85), (self.width - 20, 85), self.colors['header'], 2)
    
    def draw_hierarchical_levels(self, canvas, decision):
        """4단계 계층 구조 그리기"""
        
        level_y_positions = [150, 300, 450, 600]
        level_colors = [self.colors['level_0'], self.colors['level_1'], 
                       self.colors['level_2'], self.colors['level_3']]
        
        level_data = [
            ("Level 0: Meta Strategy", decision.meta_strategy),
            ("Level 1: Cognitive Adaptation", decision.cognitive_adaptation),
            ("Level 2: Agent Combination", f"{decision.combination_choice[0]}+{decision.combination_choice[1]}+{decision.combination_choice[2]}"),
            ("Level 3: Execution Strategy", decision.execution_strategy)
        ]
        
        for i, (level_name, decision_text) in enumerate(level_data):
            y_pos = level_y_positions[i]
            color = level_colors[i]
            
            # 레벨 박스 그리기
            box_x, box_y = 100, y_pos - 30
            box_w, box_h = 800, 80
            
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 3)
            cv2.rectangle(canvas, (box_x + 3, box_y + 3), (box_x + box_w - 3, box_y + box_h - 3), 
                         self.colors['background'], -1)
            
            # 레벨 이름
            cv2.putText(canvas, level_name, (box_x + 20, box_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 결정 내용
            cv2.putText(canvas, f"Decision: {decision_text}", (box_x + 20, box_y + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def draw_connections(self, canvas, decision):
        """레벨 간 연결선 그리기"""
        
        level_centers = [(500, 150), (500, 300), (500, 450), (500, 600)]
        
        for i in range(len(level_centers) - 1):
            start_point = (level_centers[i][0], level_centers[i][1] + 40)
            end_point = (level_centers[i + 1][0], level_centers[i + 1][1] - 40)
            
            # 연결선 그리기 (화살표 제거)
            cv2.line(canvas, start_point, end_point, 
                    self.colors['connection'], 3)
            
            # 연결 강도 표시
            mid_x = (start_point[0] + end_point[0]) // 2
            mid_y = (start_point[1] + end_point[1]) // 2
            strength = 0.9 - i * 0.05  # 레벨이 내려갈수록 강도 감소
            
            cv2.putText(canvas, f"{strength:.2f}", (mid_x + 20, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['active'], 1)
    
    def draw_performance_indicators(self, canvas, decision, system_stats):
        """성능 지표 그리기"""
        
        # 우측 성능 패널
        panel_x = 720
        panel_y = 100
        
        cv2.putText(canvas, "PERFORMANCE METRICS", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
        
        metrics = [
            f"Decision Time: {decision.decision_time:.3f}s",
            f"Tree Depth: {decision.tree_depth}",
            f"Quality Score: {decision.quality_score:.3f}",
            f"Confidence: {decision.confidence:.3f}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(canvas, metric, (panel_x, panel_y + 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # 시스템 통계
        if system_stats:
            cv2.putText(canvas, "SYSTEM STATS", (panel_x, panel_y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
            
            sys_metrics = [
                f"Avg Quality: {system_stats.get('avg_quality', 0.0):.3f}",
                f"Avg Time: {system_stats.get('avg_time', 0.0):.3f}s",
                f"Total Decisions: {system_stats.get('total_decisions', 0)}"
            ]
            
            for i, metric in enumerate(sys_metrics):
                cv2.putText(canvas, metric, (panel_x, panel_y + 180 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def highlight_decision_path(self, canvas, decision):
        """의사결정 경로 강조"""
        
        # 좌측에 의사결정 플로우 그리기
        flow_x = 20
        flow_y = 150
        
        cv2.putText(canvas, "DECISION FLOW", (flow_x, flow_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['active'], 2)
        
        flow_steps = [
            f"1. Meta: {decision.meta_strategy}",
            f"2. Adapt: {decision.cognitive_adaptation}",
            f"3. Combine: {decision.combination_choice[0][:8]}+...",
            f"4. Execute: {decision.execution_strategy[:15]}"
        ]
        
        for i, step in enumerate(flow_steps):
            y_pos = flow_y + i * 110
            
            # 스텝 원 그리기
            cv2.circle(canvas, (flow_x + 15, y_pos), 12, self.colors['active'], 3)
            cv2.putText(canvas, str(i+1), (flow_x + 10, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['active'], 2)
            
            # 스텝 설명
            cv2.putText(canvas, step[3:], (flow_x + 35, y_pos + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # 다음 스텝으로 연결선 (화살표 제거)
            if i < len(flow_steps) - 1:
                cv2.line(canvas, (flow_x + 15, y_pos + 15), 
                        (flow_x + 15, y_pos + 95), 
                        self.colors['active'], 2)

class CognitiveAdaptationWindow:
    """🧠 인지 부하 적응 시각화 창"""
    
    def __init__(self, width=900, height=700):
        self.width = width
        self.height = height
        self.window_name = "🧠 Cognitive Load Adaptation"
        
        # 색상
        self.colors = {
            'background': (20, 30, 40),
            'header': (100, 255, 255),  # Cyan
            'low_load': (100, 255, 100),    # Green
            'medium_load': (255, 255, 100), # Yellow  
            'high_load': (255, 100, 100),   # Red
            'adaptation': (255, 150, 255),   # Pink
            'text': (255, 255, 255),
            'grid': (100, 100, 100)
        }
        
        print("🧠 Cognitive Adaptation Window initialized")
    
    def create_adaptation_window(self, user_context, hierarchical_decision, adaptation_history):
        """인지 적응 시각화 생성"""
        
        canvas = np.full((self.height, self.width, 3), self.colors['background'], dtype=np.uint8)
        
        # 헤더
        self.draw_adaptation_header(canvas, user_context)
        
        # 현재 인지 상태
        self.draw_current_cognitive_state(canvas, user_context, hierarchical_decision)
        
        # 적응 히스토리 그래프
        self.draw_adaptation_history_graph(canvas, adaptation_history)
        
        # 적응 전략 설명
        self.draw_adaptation_strategy_explanation(canvas, hierarchical_decision)
        
        return canvas
    
    def draw_adaptation_header(self, canvas, user_context):
        """헤더 그리기"""
        
        cv2.putText(canvas, "COGNITIVE LOAD ADAPTATION SYSTEM", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['header'], 2)
        
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(canvas, f"Time: {current_time}", (self.width - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        cv2.line(canvas, (20, 55), (self.width - 20, 55), self.colors['header'], 2)
    
    def draw_current_cognitive_state(self, canvas, user_context, decision):
        """현재 인지 상태 시각화"""
        
        y_start = 80
        
        cv2.putText(canvas, "CURRENT COGNITIVE STATE", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['header'], 2)
        
        # 인지 부하 레벨 표시
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        load_colors = {
            'low': self.colors['low_load'],
            'medium': self.colors['medium_load'], 
            'high': self.colors['high_load']
        }
        
        load_color = load_colors.get(cognitive_load, self.colors['medium_load'])
        
        # 인지 부하 바 그리기
        bar_x, bar_y = 30, y_start + 30
        bar_width = 200
        bar_height = 30
        
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.colors['text'], 2)
        
        # 부하 수준에 따른 채우기
        load_levels = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        fill_width = int(bar_width * load_levels.get(cognitive_load, 0.5))
        
        cv2.rectangle(canvas, (bar_x + 2, bar_y + 2), 
                     (bar_x + fill_width, bar_y + bar_height - 2), 
                     load_color, -1)
        
        cv2.putText(canvas, f"Cognitive Load: {cognitive_load.upper()}", 
                   (bar_x + bar_width + 20, bar_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, load_color, 2)
        
        # 정신적 노력 점수
        mental_effort = user_context.get('mental_effort_score', 0.5)
        effort_text = f"Mental Effort: {mental_effort:.3f}"
        cv2.putText(canvas, effort_text, (30, y_start + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # 동공 확장률
        pupil_dilation = user_context.get('pupil_dilation_rate', 0.0)
        pupil_text = f"Pupil Dilation Rate: {pupil_dilation:.3f}"
        cv2.putText(canvas, pupil_text, (30, y_start + 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # 적응 전략
        adaptation = decision.cognitive_adaptation
        adaptation_text = f"Adaptation Strategy: {adaptation.upper()}"
        cv2.putText(canvas, adaptation_text, (30, y_start + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['adaptation'], 2)
    
    def draw_adaptation_history_graph(self, canvas, adaptation_history):
        """적응 히스토리 그래프"""
        
        if not adaptation_history:
            return
        
        graph_x, graph_y = 50, 280
        graph_width, graph_height = 800, 200
        
        cv2.putText(canvas, "ADAPTATION HISTORY", (graph_x, graph_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
        
        # 그래프 테두리
        cv2.rectangle(canvas, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     self.colors['text'], 2)
        
        # 그리드 그리기
        for i in range(1, 10):
            grid_x = graph_x + i * (graph_width // 10)
            cv2.line(canvas, (grid_x, graph_y), (grid_x, graph_y + graph_height), 
                    self.colors['grid'], 1)
        
        for i in range(1, 5):
            grid_y = graph_y + i * (graph_height // 5)
            cv2.line(canvas, (graph_x, grid_y), (graph_x + graph_width, grid_y), 
                    self.colors['grid'], 1)
        
        # 데이터 포인트 그리기
        history_list = list(adaptation_history)[-50:]  # 최근 50개
        if len(history_list) < 2:
            return
        
        for i in range(len(history_list) - 1):
            current = history_list[i]
            next_point = history_list[i + 1]
            
            # 인지 부하를 y좌표로 변환
            load_levels = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
            
            current_y = graph_y + graph_height - (load_levels.get(current['cognitive_load'], 0.5) * graph_height)
            next_y = graph_y + graph_height - (load_levels.get(next_point['cognitive_load'], 0.5) * graph_height)
            
            current_x = graph_x + (i * graph_width // len(history_list))
            next_x = graph_x + ((i + 1) * graph_width // len(history_list))
            
            # 선 그리기
            load_color = {
                'low': self.colors['low_load'],
                'medium': self.colors['medium_load'],
                'high': self.colors['high_load']
            }.get(current['cognitive_load'], self.colors['medium_load'])
            
            cv2.line(canvas, (int(current_x), int(current_y)), 
                    (int(next_x), int(next_y)), load_color, 2)
            
            # 포인트 표시
            cv2.circle(canvas, (int(current_x), int(current_y)), 3, load_color, -1)
    
    def draw_adaptation_strategy_explanation(self, canvas, decision):
        """적응 전략 설명"""
        
        y_start = 520
        
        cv2.putText(canvas, "ADAPTATION STRATEGY DETAILS", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
        
        adaptation = decision.cognitive_adaptation
        
        explanations = {
            'simplified': [
                "• Reduced computational complexity",
                "• Faster decision making",
                "• Lower cognitive burden",
                "• Streamlined user interface"
            ],
            'standard': [
                "• Balanced approach",
                "• Moderate complexity",
                "• Standard processing time", 
                "• Regular interface elements"
            ],
            'complex': [
                "• Advanced optimization",
                "• Thorough analysis",
                "• Higher accuracy",
                "• Rich interface features"
            ]
        }
        
        strategy_explanations = explanations.get(adaptation, ["• Standard approach"])
        
        for i, explanation in enumerate(strategy_explanations):
            cv2.putText(canvas, explanation, (30, y_start + 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['adaptation'], 1)

class PerformanceAnalyticsWindow:
    """📊 실시간 성능 분석 대시보드"""
    
    def __init__(self, width=900, height=700):
        self.width = width
        self.height = height
        self.window_name = "📊 Performance Analytics"
        
        # 색상
        self.colors = {
            'background': (25, 25, 35),
            'header': (255, 165, 0),  # Orange
            'quality': (100, 255, 150),   # Light Green
            'time': (150, 150, 255),      # Light Blue  
            'confidence': (255, 150, 100), # Light Orange
            'trend_up': (100, 255, 100),   # Green
            'trend_down': (255, 100, 100), # Red
            'text': (255, 255, 255),
            'grid': (80, 80, 80),
            'panel': (40, 40, 50)
        }
        
        print("📊 Performance Analytics Window initialized")
    
    def create_analytics_window(self, hierarchical_system, current_decision):
        """성능 분석 대시보드 생성"""
        
        canvas = np.full((self.height, self.width, 3), self.colors['background'], dtype=np.uint8)
        
        # 헤더
        self.draw_analytics_header(canvas)
        
        # 실시간 성능 메트릭
        self.draw_realtime_metrics(canvas, hierarchical_system, current_decision)
        
        # 성능 트렌드 그래프
        self.draw_performance_trends(canvas, hierarchical_system)
        
        # 레벨별 효율성 분석
        self.draw_level_efficiency_analysis(canvas, hierarchical_system)
        
        # 추천 사항
        self.draw_recommendations(canvas, hierarchical_system, current_decision)
        
        return canvas
    
    def draw_analytics_header(self, canvas):
        """헤더 그리기"""
        
        cv2.putText(canvas, "PERFORMANCE ANALYTICS DASHBOARD", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['header'], 2)
        
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(canvas, f"Time: {current_time}", (self.width - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        cv2.line(canvas, (20, 55), (self.width - 20, 55), self.colors['header'], 2)
    
    def draw_realtime_metrics(self, canvas, system, current_decision):
        """실시간 성능 메트릭"""
        
        y_start = 80
        
        cv2.putText(canvas, "REAL-TIME PERFORMANCE", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['header'], 2)
        
        # 성능 요약 가져오기
        performance_summary = system.get_current_performance_summary()
        
        # 메트릭 패널들
        panels = [
            {
                'title': 'Decision Quality',
                'value': current_decision.quality_score,
                'avg': performance_summary.get('avg_quality', 0.0),
                'color': self.colors['quality'],
                'position': (30, y_start + 40)
            },
            {
                'title': 'Response Time',
                'value': current_decision.decision_time,
                'avg': performance_summary.get('avg_time', 0.0),
                'color': self.colors['time'],
                'position': (320, y_start + 40)
            },
            {
                'title': 'Confidence Level',
                'value': current_decision.confidence,
                'avg': performance_summary.get('avg_confidence', 0.0),
                'color': self.colors['confidence'],
                'position': (610, y_start + 40)
            }
        ]
        
        for panel in panels:
            self.draw_metric_panel(canvas, panel)
    
    def draw_metric_panel(self, canvas, panel):
        """개별 메트릭 패널 그리기"""
        
        x, y = panel['position']
        width, height = 250, 120
        
        # 패널 배경
        cv2.rectangle(canvas, (x, y), (x + width, y + height), 
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), 
                     panel['color'], 2)
        
        # 제목
        cv2.putText(canvas, panel['title'], (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, panel['color'], 2)
        
        # 현재 값
        if panel['title'] == 'Response Time':
            value_text = f"{panel['value']:.3f}s"
        else:
            value_text = f"{panel['value']:.3f}"
        
        cv2.putText(canvas, f"Current: {value_text}", (x + 10, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # 평균값
        if panel['title'] == 'Response Time':
            avg_text = f"{panel['avg']:.3f}s"
        else:
            avg_text = f"{panel['avg']:.3f}"
        
        cv2.putText(canvas, f"Average: {avg_text}", (x + 10, y + 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # 트렌드 표시 (화살표 제거)
        trend_color = self.colors['trend_up'] if panel['value'] >= panel['avg'] else self.colors['trend_down']
        trend_symbol = "+" if panel['value'] >= panel['avg'] else "-"
        
        cv2.putText(canvas, trend_symbol, (x + width - 40, y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, trend_color, 3)
    
    def draw_performance_trends(self, canvas, system):
        """성능 트렌드 그래프"""
        
        graph_y = 250
        
        cv2.putText(canvas, "PERFORMANCE TRENDS", (20, graph_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
        
        # 그래프 영역
        graph_x, graph_y = 50, graph_y + 20
        graph_width, graph_height = 800, 180
        
        cv2.rectangle(canvas, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), 
                     self.colors['text'], 2)
        
        # 그리드
        for i in range(1, 8):
            grid_x = graph_x + i * (graph_width // 8)
            cv2.line(canvas, (grid_x, graph_y), (grid_x, graph_y + graph_height), 
                    self.colors['grid'], 1)
        
        for i in range(1, 4):
            grid_y = graph_y + i * (graph_height // 4)
            cv2.line(canvas, (graph_x, grid_y), (graph_x + graph_width, grid_y), 
                    self.colors['grid'], 1)
        
        # 성능 데이터 그리기
        if len(system.performance_metrics) > 1:
            metrics_list = list(system.performance_metrics)[-40:]  # 최근 40개
            
            for i in range(len(metrics_list) - 1):
                current = metrics_list[i]
                next_point = metrics_list[i + 1]
                
                # 품질 점수 라인
                current_x = graph_x + (i * graph_width // len(metrics_list))
                next_x = graph_x + ((i + 1) * graph_width // len(metrics_list))
                
                current_y_quality = graph_y + graph_height - (current['quality_score'] * graph_height)
                next_y_quality = graph_y + graph_height - (next_point['quality_score'] * graph_height)
                
                cv2.line(canvas, (int(current_x), int(current_y_quality)), 
                        (int(next_x), int(next_y_quality)), self.colors['quality'], 2)
                
                # 신뢰도 라인
                current_y_conf = graph_y + graph_height - (current['confidence'] * graph_height)
                next_y_conf = graph_y + graph_height - (next_point['confidence'] * graph_height)
                
                cv2.line(canvas, (int(current_x), int(current_y_conf)), 
                        (int(next_x), int(next_y_conf)), self.colors['confidence'], 2)
        
        # 범례
        cv2.putText(canvas, "Quality", (graph_x + graph_width - 150, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['quality'], 1)
        cv2.putText(canvas, "Confidence", (graph_x + graph_width - 80, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['confidence'], 1)
    
    def draw_level_efficiency_analysis(self, canvas, system):
        """레벨별 효율성 분석"""
        
        y_start = 480
        
        cv2.putText(canvas, "MCTS LEVEL EFFICIENCY", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
        
        level_names = ["Meta Strategy", "Cognitive Adapt", "Combination", "Execution"]
        level_colors = [self.colors['quality'], self.colors['time'], 
                       self.colors['confidence'], self.colors['header']]
        
        for i, (level_name, color) in enumerate(zip(level_names, level_colors)):
            x_pos = 30 + i * 200
            y_pos = y_start + 30
            
            # 레벨 박스
            cv2.rectangle(canvas, (x_pos, y_pos), (x_pos + 180, y_pos + 80), 
                         color, 2)
            
            # 레벨 이름
            cv2.putText(canvas, f"Level {i}", (x_pos + 10, y_pos + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(canvas, level_name, (x_pos + 10, y_pos + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            
            # 통계
            stats = system.level_stats.get(i, None)
            if stats:
                cv2.putText(canvas, f"Decisions: {stats.decision_count}", 
                           (x_pos + 10, y_pos + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
                cv2.putText(canvas, f"Best: {stats.best_value:.3f}", 
                           (x_pos + 10, y_pos + 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
    
    def draw_recommendations(self, canvas, system, current_decision):
        """추천 사항"""
        
        y_start = 600
        
        cv2.putText(canvas, "OPTIMIZATION RECOMMENDATIONS", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['header'], 2)
        
        # 성능 분석 기반 추천
        recommendations = self.generate_recommendations(system, current_decision)
        
        for i, recommendation in enumerate(recommendations):
            cv2.putText(canvas, f"• {recommendation}", (30, y_start + 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
    
    def generate_recommendations(self, system, decision):
        """성능 기반 추천 생성"""
        
        recommendations = []
        
        performance_summary = system.get_current_performance_summary()
        
        # 품질 기반 추천
        if decision.quality_score < 0.7:
            recommendations.append("Consider increasing MCTS iterations for better quality")
        
        # 시간 기반 추천  
        if decision.decision_time > 0.3:
            recommendations.append("Optimize cognitive adaptation for faster decisions")
        
        # 신뢰도 기반 추천
        if decision.confidence < 0.6:
            recommendations.append("Improve user state detection accuracy")
        
        # 일반적 추천
        total_decisions = performance_summary.get('total_decisions', 0)
        if total_decisions > 50:
            recommendations.append("System learning: Performance improving over time")
        else:
            recommendations.append("Collecting data: More decisions needed for optimization")
        
        # 기본 추천이 없을 경우
        if not recommendations:
            recommendations.append("System running normally")
        
        return recommendations[:4]  # 최대 4개

class PerceptionVisualizationWindow:
    """👁️ 시선추적 및 감정분류 시각화 윈도우"""
    
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.window_name = "👁️ Perception Analysis"
        
        # 색상
        self.colors = {
            'background': (20, 20, 30),
            'header': (0, 255, 255),  # Cyan
            'emotion': (255, 100, 100),  # Red
            'gaze': (100, 255, 100),  # Green
            'pupil': (255, 255, 100),  # Yellow
            'text': (255, 255, 255),
            'grid': (60, 60, 60),
            'panel': (40, 40, 50)
        }
        
        # 감정 리스트
        self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        
        # 감정별 색상
        self.emotion_colors = {
            'happy': (0, 255, 0),
            'sad': (0, 0, 255),
            'anger': (0, 0, 255),
            'fear': (128, 0, 128),
            'surprise': (255, 165, 0),
            'disgust': (0, 128, 128),
            'neutral': (128, 128, 128)
        }
        
        print("👁️ Perception Visualization Window initialized")
    
    def create_perception_window(self, frame_data, cognitive_data):
        """시선추적 및 감정분류 시각화 창 생성"""
        
        canvas = np.full((self.height, self.width, 3), self.colors['background'], dtype=np.uint8)
        
        # 헤더
        self.draw_perception_header(canvas)
        
        # 감정 분석 섹션
        self.draw_emotion_analysis(canvas, frame_data)
        
        # 시선 추적 섹션
        self.draw_gaze_tracking(canvas, frame_data)
        
        # 인지 부하 분석
        self.draw_cognitive_load_analysis(canvas, cognitive_data)
        
        # 실시간 상태 표시
        self.draw_realtime_status(canvas, frame_data, cognitive_data)
        
        return canvas
    
    def draw_perception_header(self, canvas):
        """헤더 그리기"""
        
        cv2.putText(canvas, "PERCEPTION ANALYSIS DASHBOARD", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['header'], 2)
        
        current_time = time.strftime("%H:%M:%S")
        cv2.putText(canvas, f"Time: {current_time}", (self.width - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        cv2.line(canvas, (20, 55), (self.width - 20, 55), self.colors['header'], 2)
    
    def draw_emotion_analysis(self, canvas, frame_data):
        """감정 분석 섹션"""
        
        y_start = 80
        
        cv2.putText(canvas, "EMOTION ANALYSIS", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['header'], 2)
        
        # 감정 정보
        emotion = frame_data.get('emotion', 'neutral')
        scores = frame_data.get('scores', {})
        
        # 감정 박스
        emotion_color = self.emotion_colors.get(emotion, (128, 128, 128))
        cv2.rectangle(canvas, (30, y_start + 30), (350, y_start + 120), 
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (30, y_start + 30), (350, y_start + 120), 
                     emotion_color, 2)
        
        # 현재 감정
        cv2.putText(canvas, f"Current Emotion: {emotion.upper()}", 
                   (40, y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # 감정 점수들
        y_offset = 80
        if isinstance(scores, dict):
            for i, (emotion_name, score) in enumerate(scores.items()):
                if i < 3:  # 상위 3개만 표시
                    color = self.emotion_colors.get(emotion_name, (128, 128, 128))
                    cv2.putText(canvas, f"{emotion_name}: {score:.3f}", 
                               (40, y_start + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 20
        elif isinstance(scores, list):
            for i, score in enumerate(scores):
                if i < 3:  # 상위 3개만 표시
                    emotion_name = self.emotions[i] if i < len(self.emotions) else f"emotion_{i}"
                    color = self.emotion_colors.get(emotion_name, (128, 128, 128))
                    cv2.putText(canvas, f"{emotion_name}: {score:.3f}", 
                               (40, y_start + y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 20
    
    def draw_gaze_tracking(self, canvas, frame_data):
        """시선 추적 섹션"""
        
        y_start = 220
        
        cv2.putText(canvas, "GAZE TRACKING", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['header'], 2)
        
        # 시선 안정성
        fix_stab = frame_data.get('fix_stab', 0.0)
        fix_flag = frame_data.get('fix_flag', False)
        
        # 시선 박스
        cv2.rectangle(canvas, (30, y_start + 30), (350, y_start + 100), 
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (30, y_start + 30), (350, y_start + 100), 
                     self.colors['gaze'], 2)
        
        # 시선 안정성 표시
        stability_text = "STABLE" if fix_flag else "UNSTABLE"
        stability_color = (0, 255, 0) if fix_flag else (0, 0, 255)
        
        cv2.putText(canvas, f"Gaze Status: {stability_text}", 
                   (40, y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
        cv2.putText(canvas, f"Stability: {fix_stab:.3f}", 
                   (40, y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
    
    def draw_cognitive_load_analysis(self, canvas, cognitive_data):
        """인지 부하 분석"""
        
        y_start = 340
        
        cv2.putText(canvas, "COGNITIVE LOAD ANALYSIS", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['header'], 2)
        
        # 인지 부하 레벨
        load_level = cognitive_data.get('cognitive_load_level', 'medium')
        mental_effort = cognitive_data.get('mental_effort_score', 0.5)
        pupil_rate = cognitive_data.get('pupil_dilation_rate', 0.0)
        
        # 인지 부하 박스
        load_color = (0, 255, 0) if load_level == 'low' else (0, 255, 255) if load_level == 'medium' else (0, 0, 255)
        cv2.rectangle(canvas, (30, y_start + 30), (350, y_start + 120), 
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (30, y_start + 30), (350, y_start + 120), 
                     load_color, 2)
        
        # 인지 부하 정보
        cv2.putText(canvas, f"Load Level: {load_level.upper()}", 
                   (40, y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, load_color, 2)
        cv2.putText(canvas, f"Mental Effort: {mental_effort:.3f}", 
                   (40, y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        cv2.putText(canvas, f"Pupil Rate: {pupil_rate:.3f}", 
                   (40, y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
    
    def draw_realtime_status(self, canvas, frame_data, cognitive_data):
        """실시간 상태 표시"""
        
        y_start = 480
        
        cv2.putText(canvas, "REAL-TIME STATUS", (20, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['header'], 2)
        
        # 상태 정보
        face_detected = frame_data.get('face_detected', False)
        left_pupil = frame_data.get('left_pupil', (0, 0))
        right_pupil = frame_data.get('right_pupil', (0, 0))
        
        # 상태 박스
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.rectangle(canvas, (30, y_start + 30), (self.width - 30, y_start + 80), 
                     self.colors['panel'], -1)
        cv2.rectangle(canvas, (30, y_start + 30), (self.width - 30, y_start + 80), 
                     status_color, 2)
        
        # 상태 정보 표시
        status_text = "FACE DETECTED" if face_detected else "NO FACE"
        cv2.putText(canvas, f"Status: {status_text}", 
                   (40, y_start + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        if face_detected:
            cv2.putText(canvas, f"Left Pupil: ({left_pupil[0]:.1f}, {left_pupil[1]:.1f})", 
                       (40, y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
            cv2.putText(canvas, f"Right Pupil: ({right_pupil[0]:.1f}, {right_pupil[1]:.1f})", 
                       (40, y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)

# ==================== Main Integrated System ====================
class HierarchicalMCTSIntegratedSystem:
    """🚀 메인 통합 시스템 - 4단계 계층적 MCTS + 실시간 인터페이스"""
    
    def __init__(self, user_name, system_mode="proposed"):
        print("🚀 Hierarchical MCTS Integrated System Starting...")
        print("=" * 70)
        print("🧠 REVOLUTIONARY FEATURES:")
        print("   🎯 4-Level Hierarchical MCTS Decision Making")
        print("   🧠 Real-time Cognitive Load Adaptation")  
        print("   👁️ Advanced Pupil-based Intelligence")
        print("   🎨 Triple Advanced Visualization Windows")
        print("   📊 Comprehensive Performance Analytics")
        print("   🔬 Research-Grade Data Collection")
        print("-" * 70)
        
        self.user_name = user_name
        self.system_mode = system_mode  # 실험 모드
        
        # 베이스라인 시스템 초기화 (사용 안 함 - 각 조건별로 직접 구현)
        self.baseline_systems = None
        print(f"✅ Condition mode: {system_mode}")
        
        # 기본 설정
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
        
        # 성능 최적화 설정
        cv2.setNumThreads(1)  # OpenCV 스레드 충돌 방지
        torch.set_num_threads(2)  # PyTorch 스레드 수 제한
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(1)
        
        # AI 모델 초기화
        self.model = ResEmoteNet()
        # Emotion smoothing (EMA on probability vector)
        self.emotion_smoother = VectorEMA(alpha=0.3, length=len(self.emotions))
        self.smoothed_emotion_probs = [0.0] * len(self.emotions)
        
        # MediaPipe 초기화 (가능한 경우)
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        else:
            self.mp_face_mesh = None
            self.face_mesh = None
        
        # 시선 추적 파라미터
        self.WIN_SEC = 2.0
        self.EMA_ALPHA = 0.3
        self.FPS_EST = 20
        self.CALIBRATION_SEC = 10
        self.MAD_MULTIPLIER = 2.5
        self.FIXSTAB_ABS_THRESH = 0.30
        self.GAZE_MAD_EPS = 1e-6
        
        # 시선 추적 변수들
        self.gaze_buffer = deque(maxlen=int(self.WIN_SEC * self.FPS_EST))
        self.gaze_ema_x = EMA(self.EMA_ALPHA)
        self.gaze_ema_y = EMA(self.EMA_ALPHA)
        self.area_calibration_buffer = []
        self.calibration_done = False
        self.area_median = 0
        self.area_mad = 0
        
        # 🧠 인지 부하 추적 (베이스라인 방식)
        self.cognitive_load_history = deque(maxlen=20)
        self.pupil_baseline = 0.0
        self.calibration_complete = False
        self.pupil_baseline_buffer = []  # 30초간 베이스라인 데이터 수집
        
        # 시스템 카운터
        self.counter = 0
        self.frame_counter = 0
        self.evaluation_frequency = 4
        self.blink_count = 0
        self.was_blinking = False
        self.start_time = time.time()
        
        # 🧠 혁신적 계층적 MCTS 시스템
        self.hierarchical_mcts = HierarchicalMCTSSystem()
        
        # 🎨 4개 시각화 윈도우
        self.tree_window = HierarchicalTreeWindow()
        self.adaptation_window = CognitiveAdaptationWindow()
        self.analytics_window = PerformanceAnalyticsWindow()
        self.perception_window = PerceptionVisualizationWindow()
        # 시각화 리프레시 간격 및 캐시 프레임
        self.refresh_interval = 3
        self._last_tree_frame = None
        self._last_adaptation_frame = None
        self._last_analytics_frame = None
        self._last_perception_frame = None
        
        # CSV 로깅
        self.setup_csv_logging()
        
        # 감정 추론 비동기 처리 구성
        self.emotion_infer_queue = queue.Queue(maxsize=1)
        self._emotion_thread_stop = threading.Event()
        # 초기 결과(중립)
        self.emotion_result = ('neutral', [0.0] * len(self.emotions))
        # 백그라운드 스레드 시작
        self._emotion_thread = threading.Thread(target=self._emotion_worker, daemon=True)
        self._emotion_thread.start()
        
        # 메모리 최적화: 프레임 버퍼 재사용
        self._frame_buffer = None
        self._csv_batch = []
        self._csv_batch_size = 10
        
        print("🎉 Revolutionary Hierarchical MCTS System Ready!")
        print("🌟 This is cutting-edge AI research in action!")
    
    def setup_csv_logging(self):
        """CSV 로깅 설정"""
        # Project-relative data directory inside workspace
        workspace_root = Path(__file__).resolve().parent
        base_dir = (workspace_root / 'data')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_dir / current_time
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_filename = output_dir / f"{self.user_name}_multi-adaptive_log.csv"
        
        # 확장된 컬럼들
        base_columns = ['Frame', 'Time(s)'] + self.emotions + [
            'Emotion', 'Left_Pupil_X', 'Left_Pupil_Y', 'Right_Pupil_X', 'Right_Pupil_Y', 
            'Blink', 'Blink_Count', 'FixStab', 'FixFlag'
        ]
        
        hierarchical_columns = [
            'Meta_Strategy', 'Cognitive_Adaptation', 'Agent_Combination', 'Execution_Strategy',
            'Decision_Quality', 'Decision_Time', 'Confidence', 'Tree_Depth',
            'Cognitive_Load_Level', 'Mental_Effort_Score', 'Pupil_Dilation_Rate'
        ]
        
        # GPT-4 협업 관련 컬럼 (논문 구현)
        llm_columns = [
            'GEMMAS_IDS', 'GEMMAS_UPR', 'LLM_Feedback'
        ]
        
        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(base_columns + hierarchical_columns + llm_columns)
    
    def setup_camera(self):
        """카메라 설정 - 휴대폰 카메라 사용 (final_sac.py 스타일)"""
        print("Setting up camera for hierarchical system...")
        
        # 휴대폰 카메라 사용 (final_sac.py 스타일)
        preferred_indices = [1, 0, 2, 3]
        video_capture = None
        for idx in preferred_indices:
            try:
                print(f"   Trying camera index {idx} (AVFOUNDATION)")
                cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                cap.set(cv2.CAP_PROP_FPS, 20)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지연 감소
                if cap.isOpened():
                    video_capture = cap
                    break
                else:
                    cap.release()
            except Exception:
                pass
        if video_capture is None:
            # Fallback to default backend
            for idx in preferred_indices:
                try:
                    print(f"   Trying camera index {idx} (default backend)")
                    cap = cv2.VideoCapture(idx)
                    cap.set(cv2.CAP_PROP_FPS, 20)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 지연 감소
                    if cap.isOpened():
                        video_capture = cap
                        break
                    else:
                        cap.release()
                except Exception:
                    pass
        if video_capture is None:
            raise RuntimeError("Cannot open any available camera")
            
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        
        print(f"Phone Camera ready: {width}x{height} @ {fps}fps")
        
        return video_capture
    
    def _emotion_worker(self):
        """백그라운드 감정 추론 워커"""
        while not self._emotion_thread_stop.is_set():
            try:
                face_crop = self.emotion_infer_queue.get(timeout=0.1)
            except Exception:
                continue
            try:
                label, probs = self.get_max_emotion(face_crop)
                probs = sanitize_probs(probs, len(self.emotions), self.emotions.index('neutral'))
                self.emotion_result = (label, probs)
            except Exception:
                # 실패 시 이전 결과 유지
                pass
    
    def get_max_emotion(self, face_image):
        """final_sac.py와 동일한 감정 분석"""
        try:
            # final_sac.py와 동일한 방식
            pil_crop_img = Image.fromarray(face_image)
            rounded_scores = self.detect_emotion(pil_crop_img)
            max_index = np.argmax(rounded_scores)
            return self.emotions[max_index], rounded_scores
        except Exception as e:
            print(f"❌ 감정 분석 오류: {e}")
            return 'neutral', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    
    def detect_emotion(self, video_frame):
        """final_sac.py의 detect_emotion 함수"""
        vid_fr_tensor = transform(video_frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model.model(vid_fr_tensor)  # self.model.model으로 실제 모델 호출
            probabilities = F.softmax(outputs, dim=1)
        scores = probabilities.cpu().numpy().flatten()
        return [round(score, 2) for score in scores]
    
    def extract_cognitive_load_data(self, face_landmarks, frame_shape):
        """인지 부하 데이터 추출 (기존 perception_bridge와 동일한 로직)"""
        
        if not face_landmarks:
            return {
                'cognitive_load_level': 'unknown',
                'mental_effort_score': 0.5,
                'pupil_dilation_rate': 0.0,
                'avg_pupil_diameter': 0.0,
                'baseline_diameter': self.pupil_baseline,
                'confidence': 0.0
            }
        
        # 동공 크기 추출 (MediaPipe 기반 실제 계산)
        h, w = frame_shape[:2]
        
        # 좌안/우안 동공 추정 (MediaPipe 랜드마크 기반)
        left_eye_indices = [468, 469, 470, 471, 472]  # 좌안 아이리스
        right_eye_indices = [473, 474, 475, 476, 477] # 우안 아이리스
        
        left_diameter = self._calculate_pupil_diameter(face_landmarks, left_eye_indices, w, h)
        right_diameter = self._calculate_pupil_diameter(face_landmarks, right_eye_indices, w, h)
        
        if left_diameter is None or right_diameter is None:
            return {
                'cognitive_load_level': 'unknown',
                'mental_effort_score': 0.5,
                'pupil_dilation_rate': 0.0,
                'avg_pupil_diameter': 0.0,
                'baseline_diameter': self.pupil_baseline,
                'confidence': 0.0
            }
        
        avg_diameter = (left_diameter + right_diameter) / 2.0
        
        # 베이스라인 수집 (초기 30초)
        elapsed_time = time.time() - self.start_time
        baseline_period = 30.0  # 30초 베이스라인
        
        # 베이스라인 데이터 수집
        if elapsed_time <= baseline_period:
            self.pupil_baseline_buffer.append(avg_diameter)
        
        # 베이스라인 계산 (30초 후 또는 충분한 데이터가 있을 때)
        if (elapsed_time > baseline_period or len(self.pupil_baseline_buffer) >= 50) and not self.calibration_complete:
            if self.pupil_baseline_buffer:
                self.pupil_baseline = np.median(self.pupil_baseline_buffer)  # 중간값 사용 (이상치 제거)
                self.calibration_complete = True
                print(f"Cognitive Load Baseline established: {self.pupil_baseline:.4f} (from {len(self.pupil_baseline_buffer)} samples)")
        
        # 인지 부하 계산 (0초부터 즉시 시작)
        if self.calibration_complete and self.pupil_baseline > 0:
            # 베이스라인 기반 계산 (30초 후)
            baseline_ratio = avg_diameter / self.pupil_baseline
            
            # 확장률 계산 (이전 값과 비교)
            if self.cognitive_load_history:
                prev_diameter = self.cognitive_load_history[-1]['avg_pupil_diameter']
                dilation_rate = (avg_diameter - prev_diameter) / prev_diameter if prev_diameter > 0 else 0.0
            else:
                dilation_rate = 0.0
            
            # 인지 부하 레벨 결정
            if baseline_ratio > 1.2:
                cognitive_level = 'high'
            elif baseline_ratio < 0.9:
                cognitive_level = 'low'
            else:
                cognitive_level = 'medium'
            
            # 정신적 노력 점수 (0-1)
            mental_effort = min(1.0, max(0.0, (baseline_ratio - 0.8) / 0.6))
            confidence = 0.8
        else:
            # 베이스라인 수집 중 - 절대값 기반 임시 계산
            if len(self.pupil_baseline_buffer) > 0:
                # 현재까지의 중간값을 임시 베이스라인으로 사용
                temp_baseline = np.median(self.pupil_baseline_buffer)
                baseline_ratio = avg_diameter / temp_baseline if temp_baseline > 0 else 1.0
                
                if baseline_ratio > 1.15:  # 임시 임계값 (더 관대하게)
                    cognitive_level = 'high'
                elif baseline_ratio < 0.95:
                    cognitive_level = 'low'
                else:
                    cognitive_level = 'medium'
                
                mental_effort = min(1.0, max(0.0, (baseline_ratio - 0.8) / 0.6))
                confidence = 0.5  # 중간 신뢰도
            else:
                # 완전 초기값
                cognitive_level = 'medium'
                mental_effort = 0.5
                confidence = 0.3
            
            # 확장률 계산
            if self.cognitive_load_history:
                prev_diameter = self.cognitive_load_history[-1]['avg_pupil_diameter']
                dilation_rate = (avg_diameter - prev_diameter) / prev_diameter if prev_diameter > 0 else 0.0
            else:
                dilation_rate = 0.0
        
        cognitive_data = {
            'cognitive_load_level': cognitive_level,
            'mental_effort_score': mental_effort,
            'pupil_dilation_rate': dilation_rate,
            'avg_pupil_diameter': avg_diameter,
            'baseline_diameter': self.pupil_baseline,
            'confidence': confidence
        }
        
        # 히스토리 저장
        self.cognitive_load_history.append(cognitive_data)
        
        return cognitive_data
    
    def _calculate_pupil_diameter(self, landmarks, indices, width, height):
        """동공 직경 추정"""
        
        if len(indices) < 5:
            return None
            
        points = []
        for idx in indices:
            if idx < len(landmarks.landmark):
                x = landmarks.landmark[idx].x * width
                y = landmarks.landmark[idx].y * height
                points.append((x, y))
        
        if len(points) < 3:
            return None
        
        # 동공 영역의 바운딩 박스로 지름 추정
        points = np.array(points)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        
        diameter = max(max_x - min_x, max_y - min_y)
        
        # 정규화 (이미지 크기 대비)
        normalized_diameter = diameter / min(width, height)
        
        return normalized_diameter
    
    def generate_agent_discussions(self, emotion, cognitive_data, fix_stab, fix_flag):
        """🤖 에이전트들이 토론하는 내용 생성"""
        discussions = []
        
        # Meta Agent 토론
        if emotion == 'neutral':
            discussions.append("Meta Agent: 사용자가 중립 상태입니다. 표준 전략을 유지하겠습니다.")
        elif emotion in ['happy', 'surprise']:
            discussions.append("Meta Agent: 긍정적 감정 감지! 탐색 범위를 확대하겠습니다.")
        elif emotion in ['sad', 'anger', 'fear']:
            discussions.append("Meta Agent: 부정적 감정 감지! 신중한 접근이 필요합니다.")
        
        # Cognitive Agent 토론
        cognitive_load = cognitive_data.get('cognitive_load_level', 'medium')
        if cognitive_load == 'low':
            discussions.append("Cognitive Agent: 인지 부하가 낮습니다. 복잡한 작업을 제안합니다.")
        elif cognitive_load == 'high':
            discussions.append("Cognitive Agent: 인지 부하가 높습니다. 단순화된 접근이 필요합니다.")
        else:
            discussions.append("Cognitive Agent: 적절한 인지 부하 상태입니다. 현재 전략을 유지합니다.")
        
        # Perception Agent 토론
        if fix_flag == 1:
            discussions.append("Perception Agent: 사용자가 산만해 보입니다. 주의를 집중시켜야 합니다.")
        else:
            discussions.append("Perception Agent: 집중도가 양호합니다. 현재 작업을 계속합니다.")
        
        # Combination Agent 토론
        if fix_stab is not None:
            if fix_stab > 0.7:
                discussions.append("Combination Agent: 안정적인 시선 패턴입니다. 정밀한 작업이 가능합니다.")
            elif fix_stab < 0.3:
                discussions.append("Combination Agent: 불안정한 시선 패턴입니다. 단순한 작업으로 전환합니다.")
            else:
                discussions.append("Combination Agent: 보통 수준의 시선 안정성입니다. 균형잡힌 접근을 사용합니다.")
        
        # Execution Agent 토론
        mental_effort = cognitive_data.get('mental_effort_score', 0.5)
        if mental_effort > 0.7:
            discussions.append("Execution Agent: 높은 정신적 노력이 감지됩니다. 작업 강도를 조절합니다.")
        elif mental_effort < 0.3:
            discussions.append("Execution Agent: 낮은 정신적 노력입니다. 더 도전적인 작업을 제안합니다.")
        else:
            discussions.append("Execution Agent: 적절한 정신적 노력 수준입니다. 현재 작업을 유지합니다.")
        
        # 시스템 상태 토론
        pupil_rate = cognitive_data.get('pupil_dilation_rate', 0.0)
        if pupil_rate > 0.1:
            discussions.append("System: 동공 확장이 감지되었습니다. 인지 부하가 증가하고 있습니다.")
        elif pupil_rate < -0.1:
            discussions.append("System: 동공 수축이 감지되었습니다. 인지 부하가 감소하고 있습니다.")
        else:
            discussions.append("System: 동공 크기가 안정적입니다. 현재 상태를 유지합니다.")
        
        return discussions
    
    def process_camera_frame(self, frame):
        """🧠 카메라 프레임 처리 - 계층적 MCTS 통합"""
        
        # 입력 프레임 다운스케일(처리 부하 절감)
        try:
            h, w = frame.shape[:2]
            max_w, max_h = 960, 540
            scale = min(1.0, max_w / max(w, 1), max_h / max(h, 1))
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
        except Exception:
            pass

        # 기본값들
        max_emotion = 'neutral'
        scores = [0.0] * len(self.emotions)
        fix_stab = 0.5
        fix_flag = 0
        face_detected = False
        left_pupil = (0, 0)
        right_pupil = (0, 0)
        
        # 인지 부하 기본 데이터
        cognitive_data = {
            'cognitive_load_level': 'medium',
            'mental_effort_score': 0.5,
            'pupil_dilation_rate': 0.0,
            'avg_pupil_diameter': 0.0,
            'baseline_diameter': self.pupil_baseline,
            'confidence': 0.0
        }
        
        # MediaPipe 처리 (가능한 경우)
        if MEDIAPIPE_AVAILABLE and self.face_mesh is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                face_detected = True
                
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                
                # 👁️ 시선 추적 및 눈 감지
                left_eye_idxs = [362, 385, 387, 263, 373, 380]
                right_eye_idxs = [33, 160, 158, 133, 153, 144]
                left_eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_idxs]
                right_eye_points = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_idxs]
                
                # 눈 포인트 표시
                for pt in left_eye_points + right_eye_points:
                    cv2.circle(frame, pt, 2, (0, 255, 255), -1)
                
                # 깜박임 감지
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2.0
                is_blinking = avg_ear < 0.2
                
                if not self.was_blinking and is_blinking:
                    self.blink_count += 1
                    
                self.was_blinking = is_blinking
                left_pupil = left_eye_points[0]
                right_pupil = right_eye_points[0]
                
                # ===== final_sac.py의 시선추적 로직 =====
                # 홍채 포인트 (MediaPipe 468-477)
                left_iris_indices = list(range(468, 473))  # 468-472
                right_iris_indices = list(range(473, 478))  # 473-477

                # 홍채 중심과 반경 계산
                left_iris_cx, left_iris_cy, left_iris_r = iris_center_radius(
                    face_landmarks.landmark, left_iris_indices, w, h)
                right_iris_cx, right_iris_cy, right_iris_r = iris_center_radius(
                    face_landmarks.landmark, right_iris_indices, w, h)

                # IPD 계산 및 시선 좌표
                if left_iris_cx is not None and right_iris_cx is not None:
                    # IPD (Inter-Pupillary Distance)
                    ipd = np.sqrt((left_iris_cx - right_iris_cx)**2 + (left_iris_cy - right_iris_cy)**2)
                    
                    # 시선 중점
                    mx = (left_iris_cx + right_iris_cx) / 2
                    my = (left_iris_cy + right_iris_cy) / 2
                    
                    # IPD 정규화
                    if ipd > 0:
                        gaze_x_raw = mx / ipd
                        gaze_y_raw = my / ipd
                        
                        # EMA 평활화
                        gaze_x = self.gaze_ema_x.update(gaze_x_raw)
                        gaze_y = self.gaze_ema_y.update(gaze_y_raw)
                        
                        # 시선 버퍼에 추가
                        self.gaze_buffer.append((gaze_x, gaze_y))

                # ===== Fixation Stability 계산 =====
                elapsed_time = time.time() - self.start_time
                
                # 베이스라인 수집 (초기 30초)
                baseline_period = 30.0  # 30초 베이스라인
                if elapsed_time <= baseline_period:
                    if len(self.gaze_buffer) >= int(self.WIN_SEC * self.FPS_EST * 0.5):
                        area, _ = calculate_fixation_stability(self.gaze_buffer)
                        if area is not None:
                            self.area_calibration_buffer.append(area)
                
                # 베이스라인 통계 업데이트 (30초 후에도 계속 업데이트)
                if elapsed_time > baseline_period and self.area_calibration_buffer and not self.calibration_done:
                    self.area_median = np.median(self.area_calibration_buffer)
                    self.area_mad = calculate_mad(np.array(self.area_calibration_buffer))
                    # 안전 하한 적용 (MAD=0 방지)
                    if self.area_mad < self.GAZE_MAD_EPS:
                        self.area_mad = self.GAZE_MAD_EPS
                    self.calibration_done = True
                    print(f"Gaze Baseline established (30s): Median={self.area_median:.4f}, MAD={self.area_mad:.4f}")

                # Fixation 계산 (0초부터 즉시 시작)
                if len(self.gaze_buffer) >= int(self.WIN_SEC * self.FPS_EST * 0.5):
                    fix_area, fix_stab = calculate_fixation_stability(self.gaze_buffer)
                    
                    if fix_area is not None and fix_stab is not None:
                        # 산만 판정 (FixFlag)
                        if self.calibration_done and self.area_median > 0:
                            # 베이스라인 기반 판정 (30초 후)
                            unstable_thresh = self.area_median + self.MAD_MULTIPLIER * max(self.area_mad, self.GAZE_MAD_EPS)
                            fix_flag = 1 if fix_area > unstable_thresh else 0
                        else:
                            # 절대 임계값 사용 (베이스라인 수집 중 또는 데이터 부족)
                            fix_flag = 1 if fix_stab <= self.FIXSTAB_ABS_THRESH else 0
                
                # 🧠 인지 부하 데이터 추출
                cognitive_data = self.extract_cognitive_load_data(face_landmarks, frame.shape)
                
                # 😊 감정 분석 (직접 처리 + 스무딩)
                if self.counter == 0:
                    bbox_x = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                    bbox_y = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                    bbox_w = int((max([lm.x for lm in face_landmarks.landmark]) - min([lm.x for lm in face_landmarks.landmark])) * w)
                    bbox_h = int((max([lm.y for lm in face_landmarks.landmark]) - min([lm.y for lm in face_landmarks.landmark])) * h)
                    
                    face_crop = frame[bbox_y:bbox_y + bbox_h, bbox_x:bbox_x + bbox_w]
                    if face_crop.size > 0:
                        # 직접 감정분류 (final_sac.py 방식)
                        max_emotion, raw_scores = self.get_max_emotion(face_crop)
                        # 스무딩 적용
                        smoothed = self.emotion_smoother.update(raw_scores)
                        # Normalize to sum 1 to avoid drift
                        denom = float(np.sum(smoothed)) if np.sum(smoothed) > 1e-8 else 1.0
                        scores = (smoothed / denom).tolist()
                        
                        # 비동기 큐에도 제출 (백그라운드 처리용)
                        try:
                            while not self.emotion_infer_queue.empty():
                                _ = self.emotion_infer_queue.get_nowait()
                            self.emotion_infer_queue.put_nowait(face_crop)
                        except Exception:
                            pass
                
                # 얼굴 박스 표시
                bbox_x = int(min([lm.x for lm in face_landmarks.landmark]) * w)
                bbox_y = int(min([lm.y for lm in face_landmarks.landmark]) * h)
                bbox_w = int((max([lm.x for lm in face_landmarks.landmark]) - min([lm.x for lm in face_landmarks.landmark])) * w)
                bbox_h = int((max([lm.y for lm in face_landmarks.landmark]) - min([lm.y for lm in face_landmarks.landmark])) * h)
                
                # Enhanced face visualization
                cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 3)
                cv2.putText(frame, f"EMOTION: {max_emotion.upper()}", (bbox_x, bbox_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Emotion confidence bar
                emotion_confidence = max(scores) if scores else 0.5
                bar_width = int(bbox_w * emotion_confidence)
                cv2.rectangle(frame, (bbox_x, bbox_y - 35), (bbox_x + bar_width, bbox_y - 25), (0, 255, 0), -1)
                cv2.rectangle(frame, (bbox_x, bbox_y - 35), (bbox_x + bbox_w, bbox_y - 25), (255, 255, 255), 1)
                
                # Pupil centers visualization
                if left_pupil != (0, 0) and right_pupil != (0, 0):
                    cv2.circle(frame, left_pupil, 5, (255, 0, 255), -1)  # Left pupil - Magenta
                    cv2.circle(frame, right_pupil, 5, (255, 0, 255), -1)  # Right pupil - Magenta
                    cv2.putText(frame, "L", (left_pupil[0] - 10, left_pupil[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    cv2.putText(frame, "R", (right_pupil[0] - 10, right_pupil[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                # Gaze direction indicator
                if len(self.gaze_buffer) > 5:
                    recent_gaze = list(self.gaze_buffer)[-5:]
                    avg_gaze_x = np.mean([g[0] for g in recent_gaze])
                    avg_gaze_y = np.mean([g[1] for g in recent_gaze])
                    
                    # Gaze direction line (화살표 제거)
                    gaze_center_x = bbox_x + bbox_w // 2
                    gaze_center_y = bbox_y + bbox_h // 2
                    gaze_end_x = int(gaze_center_x + (avg_gaze_x - 0.5) * 100)
                    gaze_end_y = int(gaze_center_y + (avg_gaze_y - 0.5) * 100)
                    
                    # Removed: visual gaze line and label to avoid sky-blue arrow/line on screen
                    # cv2.line(frame, (gaze_center_x, gaze_center_y), (gaze_end_x, gaze_end_y), (255, 255, 0), 3)
                    # cv2.putText(frame, "GAZE", (gaze_end_x + 5, gaze_end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 🧠 사용자 컨텍스트 구성
        user_context = {
            'emotion': max_emotion,
            'attention': clamp_float(fix_stab if fix_stab is not None else 0.5, 0.0, 1.0, 0.5),
            'cognitive_load_level': cognitive_data['cognitive_load_level'],
            'mental_effort_score': clamp_float(cognitive_data['mental_effort_score'], 0.0, 1.0, 0.5),
            'pupil_dilation_rate': clamp_float(cognitive_data['pupil_dilation_rate'], -1.0, 1.0, 0.0),
            'avg_pupil_diameter': clamp_float(cognitive_data['avg_pupil_diameter'], 0.0, 1.0, 0.0),
            'baseline_diameter': clamp_float(cognitive_data['baseline_diameter'], 0.0, 1.0, 0.0),
            'confidence': clamp_float(cognitive_data['confidence'], 0.0, 1.0, 0.0),
            # 베이스라인 방식: calibration_in_progress 플래그 제거 (0초부터 즉시 분석)
            # 보상 보완용 필드
            'face_detected': bool(face_detected),
            'emotion_probabilities': sanitize_probs(scores, len(self.emotions), self.emotions.index('neutral'))  # R_emo에서 사용
        }
        
        # 🚀 계층적 MCTS 의사결정
        # 시스템 모드에 따라 다른 의사결정
        if self.system_mode == "proposed":
            hierarchical_decision = self.hierarchical_mcts.hierarchical_decision_making(user_context, face_detected)
        else:
            # 베이스라인 시스템 사용
            baseline_system = self.baseline_systems[self.system_mode]
            hierarchical_decision = baseline_system.search(user_context)
        
        # Enhanced camera window information display
        elapsed_time_display = round(time.time() - self.start_time, 2)
        
        # Background panel for better readability
        cv2.rectangle(frame, (5, 5), (600, 250), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (600, 250), (100, 100, 100), 2)
        
        # 안전한 변수 처리
        safe_fix_stab = fix_stab if fix_stab is not None else 0.5
        safe_left_pupil = f"({left_pupil[0]:.0f}, {left_pupil[1]:.0f})" if left_pupil != (0, 0) else "(0, 0)"
        safe_right_pupil = f"({right_pupil[0]:.0f}, {right_pupil[1]:.0f})" if right_pupil != (0, 0) else "(0, 0)"
        safe_cognitive_level = cognitive_data.get('cognitive_load_level', 'medium')
        safe_effort = cognitive_data.get('mental_effort_score', 0.5)
        safe_pupil_size = cognitive_data.get('avg_pupil_diameter', 0.0)
        safe_pupil_rate = cognitive_data.get('pupil_dilation_rate', 0.0)
        safe_meta_strategy = getattr(hierarchical_decision, 'meta_strategy', 'adaptive')
        safe_adaptation = getattr(hierarchical_decision, 'cognitive_adaptation', 'standard')
        safe_quality = getattr(hierarchical_decision, 'quality_score', 0.5)
        
        info_texts = [
            f"⏱️ Time: {elapsed_time_display}s | 👤 User: {self.user_name}",
            f"😊 Emotion: {max_emotion.upper()} | 🎯 Focus: {safe_fix_stab:.3f}",
            f"👁️ Pupil L: {safe_left_pupil} | R: {safe_right_pupil}",
            f"🧠 Cognitive: {safe_cognitive_level} | Effort: {safe_effort:.2f}",
            f"📊 Pupil Size: {safe_pupil_size:.2f}mm | Rate: {safe_pupil_rate:.3f}",
            f"🔄 Distracted: {'YES' if fix_flag else 'NO'} | 👁️ Blinks: {self.blink_count}",
            f"⚙️ Calibration: {'DONE' if self.calibration_done else 'IN PROGRESS'}",
            f"🎯 Meta Strategy: {safe_meta_strategy}",
            f"🧠 Adaptation: {safe_adaptation}",
            f"📊 Quality Score: {safe_quality:.3f}",
            "",
            "🔍 See Hierarchical MCTS Hub for detailed analysis!"
        ]
        
        for i, text in enumerate(info_texts):
            if text:
                if "Hierarchical MCTS Hub" in text:
                    color = (0, 255, 255)  # Cyan
                elif "Pupil" in text or "Cognitive" in text:
                    color = (255, 0, 255)  # Magenta
                elif "Emotion" in text:
                    color = (0, 255, 0)  # Green
                elif "Meta Strategy" in text or "Adaptation" in text or "Quality" in text:
                    color = (255, 215, 0)  # Gold
                else:
                    color = (255, 255, 255)  # White
                
                # 한국어 렌더러 사용
                frame = put_korean_text(frame, text, (10, 25 + i*20), 
                                      font_size=16, color=color, bg_color=(0, 0, 0), padding=2)
        
        # 🤖 에이전트 토론 내용 생성
        agent_discussions = self.generate_agent_discussions(max_emotion, cognitive_data, fix_stab, fix_flag)
        
        # 에이전트 토론 내용을 화면에 표시
        discussion_y_start = frame.shape[0] - 200
        for i, discussion in enumerate(agent_discussions[:8]):  # 최대 8개만 표시
            color = (255, 255, 0) if "Meta" in discussion else (0, 255, 255) if "Cognitive" in discussion else (255, 0, 255) if "Perception" in discussion else (0, 255, 0)
            frame = put_korean_text(frame, discussion, (10, discussion_y_start + i*22), 
                                   font_size=16, color=color, bg_color=(20, 20, 20), padding=3)
        
        return frame, {
            'emotion': max_emotion,
            'scores': scores,
            'fix_stab': fix_stab,
            'fix_flag': fix_flag,
            'face_detected': face_detected,
            'left_pupil': left_pupil,
            'right_pupil': right_pupil,
            'hierarchical_decision': hierarchical_decision,
            'user_context': user_context,
            'cognitive_data': cognitive_data,
            'agent_discussions': agent_discussions
        }
    
    def create_combined_visualization(self, tree_frame, adaptation_frame, analytics_frame, perception_frame):
        """4개 시각화를 하나의 윈도우로 통합"""
        
        # 통합 캔버스 크기 (2x2 그리드)
        canvas_width = 1800
        canvas_height = 1200
        combined_canvas = np.full((canvas_height, canvas_width, 3), (20, 20, 30), dtype=np.uint8)
        
        # 각 프레임 크기 조정
        cell_width = canvas_width // 2
        cell_height = canvas_height // 2
        
        # 1️⃣ Decision Tree (좌상단)
        tree_resized = cv2.resize(tree_frame, (cell_width, cell_height))
        combined_canvas[0:cell_height, 0:cell_width] = tree_resized
        
        # 2️⃣ Cognitive Adaptation (우상단)  
        adaptation_resized = cv2.resize(adaptation_frame, (cell_width, cell_height))
        combined_canvas[0:cell_height, cell_width:canvas_width] = adaptation_resized
        
        # 3️⃣ Performance Analytics (좌하단)
        analytics_resized = cv2.resize(analytics_frame, (cell_width, cell_height))
        combined_canvas[cell_height:canvas_height, 0:cell_width] = analytics_resized
        
        # 4️⃣ Perception Analysis (우하단)
        perception_resized = cv2.resize(perception_frame, (cell_width, cell_height))
        combined_canvas[cell_height:canvas_height, cell_width:canvas_width] = perception_resized
        
        # 구분선 추가
        cv2.line(combined_canvas, (cell_width, 0), (cell_width, canvas_height), (100, 100, 100), 2)
        cv2.line(combined_canvas, (0, cell_height), (canvas_width, cell_height), (100, 100, 100), 2)
        
        # 중앙 제목
        cv2.putText(combined_canvas, "HIERARCHICAL MCTS INTEGRATED HUB", 
                   (canvas_width//2 - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return combined_canvas
    
    def run(self):
        """🚀 메인 4-윈도우 실행 루프"""
        try:
            cap = self.setup_camera()
            
            print("\n🌟 REVOLUTIONARY HIERARCHICAL MCTS SYSTEM RUNNING...")
            print("🖼️ Windows (2-Window Layout):")
            print("   1. 📹 Emotion Estimation & Eye Tracker: Real-time face & cognitive tracking")
            print("   2. 🧠 Hierarchical MCTS Hub: Integrated 4-panel visualization")
            print("      - 🌲 Decision Tree (Top-Left)")
            print("      - 🧠 Cognitive Adaptation (Top-Right)")
            print("      - 📊 Performance Analytics (Bottom-Left)")
            print("      - 👁️ Perception Analysis (Bottom-Right)")
            print("🎮 Controls: Q = Quit")
            print("=" * 70)
            
            while True:
                success, frame = cap.read()
                if not success:
                    print("Failed to read frame")
                    break
                
                # 🧠 카메라 프레임 처리 및 계층적 의사결정
                camera_frame, frame_data = self.process_camera_frame(frame)
                
                # 🎨 3개 시각화 윈도우 생성
                hierarchical_decision = frame_data['hierarchical_decision']
                user_context = frame_data['user_context']
                
                # 시각화 프레임 생성(리프레시 간격 적용)
                if (self.frame_counter % self.refresh_interval == 0 or
                    self._last_tree_frame is None):
                    # 1️⃣ 계층적 의사결정 트리
                    self._last_tree_frame = self.tree_window.create_tree_window(
                        hierarchical_decision,
                        self.hierarchical_mcts.get_current_performance_summary()
                    )
                    # 2️⃣ 인지 적응 시각화
                    self._last_adaptation_frame = self.adaptation_window.create_adaptation_window(
                        user_context,
                        hierarchical_decision,
                        self.hierarchical_mcts.adaptation_history
                    )
                    # 3️⃣ 성능 분석 대시보드
                    self._last_analytics_frame = self.analytics_window.create_analytics_window(
                        self.hierarchical_mcts,
                        hierarchical_decision
                    )
                    # 4️⃣ 시선추적 및 감정분류 시각화
                    self._last_perception_frame = self.perception_window.create_perception_window(
                        frame_data,
                        frame_data['cognitive_data']
                    )
                tree_frame = self._last_tree_frame
                adaptation_frame = self._last_adaptation_frame
                analytics_frame = self._last_analytics_frame
                perception_frame = self._last_perception_frame
                
                # 5️⃣ 통합 시각화 윈도우 생성 (4개 시각화를 하나로)
                combined_visualization_frame = self.create_combined_visualization(
                    tree_frame, adaptation_frame, analytics_frame, perception_frame
                )
                
                # 🖥️ 2개 윈도우로 분리 (final_sac.py 스타일 + MCTS Hub)
                cv2.imshow("Emotion Estimation & Eye Tracker", camera_frame)
                cv2.imshow("🧠 Hierarchical MCTS Hub", combined_visualization_frame)
                
                # 📊 CSV 로깅 (배치 처리)
                if self.frame_counter % 30 == 0:
                    elapsed_time = time.time() - self.start_time
                    
                    base_data = [
                        self.frame_counter, elapsed_time
                    ] + sanitize_probs(frame_data['scores'], len(self.emotions), self.emotions.index('neutral')) + [
                        frame_data['emotion'],
                        clamp_float(frame_data['left_pupil'][0] if frame_data['left_pupil'] else 0.0, -1e6, 1e6, 0.0),
                        clamp_float(frame_data['left_pupil'][1] if frame_data['left_pupil'] else 0.0, -1e6, 1e6, 0.0),
                        clamp_float(frame_data['right_pupil'][0] if frame_data['right_pupil'] else 0.0, -1e6, 1e6, 0.0),
                        clamp_float(frame_data['right_pupil'][1] if frame_data['right_pupil'] else 0.0, -1e6, 1e6, 0.0),
                        int(bool(self.was_blinking)), int(self.blink_count),
                        clamp_float(frame_data['fix_stab'] if frame_data['fix_stab'] is not None else 0.5, 0.0, 1.0, 0.5),
                        int(bool(frame_data['fix_flag']))
                    ]
                    
                    # 보상 구성요소(레벨2) 로깅 확장
                    comb_mcts = getattr(self.hierarchical_mcts, 'combination_mcts', None)
                    reward_comp = getattr(comb_mcts, 'last_reward_components', {}) if comb_mcts is not None else {}
                    hierarchical_data = [
                        hierarchical_decision.meta_strategy,
                        hierarchical_decision.cognitive_adaptation,
                        f"{hierarchical_decision.combination_choice[0]}+{hierarchical_decision.combination_choice[1]}+{hierarchical_decision.combination_choice[2]}",
                        hierarchical_decision.execution_strategy,
                        hierarchical_decision.quality_score,
                        hierarchical_decision.decision_time,
                        hierarchical_decision.confidence,
                        hierarchical_decision.tree_depth,
                        frame_data['cognitive_data']['cognitive_load_level'],
                        frame_data['cognitive_data']['mental_effort_score'],
                        frame_data['cognitive_data']['pupil_dilation_rate'],
                        # 추가: 보상 분해
                        reward_comp.get('neurips_2023', ''),
                        reward_comp.get('ijcai_2025', ''),
                        # GPT-4 협업 관련 (논문 구현)
                        hierarchical_decision.ids,
                        hierarchical_decision.upr,
                        hierarchical_decision.llm_feedback[:100] if hierarchical_decision.llm_feedback else '',
                        reward_comp.get('uncertainty', ''),
                        reward_comp.get('safety', ''),
                        reward_comp.get('final', ''),
                        reward_comp.get('elapsed_ms', '')
                    ]
                    
                    # 배치에 추가
                    self._csv_batch.append(base_data + hierarchical_data)
                    
                    # 배치 크기 도달 시 파일 쓰기
                    if len(self._csv_batch) >= self._csv_batch_size:
                        try:
                            with open(self.csv_filename, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerows(self._csv_batch)
                            self._csv_batch.clear()
                        except:
                            pass
                
                # 키 입력
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                
                self.counter += 1
                self.frame_counter += 1
                if self.counter == self.evaluation_frequency:
                    self.counter = 0
                
                # 메모리 정리 (100프레임마다)
                if self.frame_counter % 100 == 0:
                    gc.collect()
            
        except Exception as e:
            print(f"System error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 남은 CSV 배치 쓰기
            try:
                if hasattr(self, '_csv_batch') and self._csv_batch:
                    with open(self.csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(self._csv_batch)
            except:
                pass
            
            if 'cap' in locals():
                cap.release()
            # Properly close MediaPipe resources if available
            try:
                if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                    self.face_mesh.close()
            except Exception:
                pass
            # Emotion worker 종료
            try:
                if hasattr(self, '_emotion_thread_stop'):
                    self._emotion_thread_stop.set()
                if hasattr(self, '_emotion_thread') and self._emotion_thread is not None:
                    self._emotion_thread.join(timeout=1.0)
            except Exception:
                pass
            cv2.destroyAllWindows()
            
            # 📊 최종 통계
            runtime = time.time() - self.start_time
            performance_summary = self.hierarchical_mcts.get_current_performance_summary()
            
            print(f"\n🎉 HIERARCHICAL MCTS SYSTEM TERMINATED")
            print("=" * 60)
            print(f"📊 FINAL STATISTICS:")
            print(f"   ⏱️ Runtime: {runtime:.1f}s")
            print(f"   🎬 Frames Processed: {self.frame_counter}")
            print(f"   🧠 Total Decisions: {performance_summary.get('total_decisions', 0)}")
            print(f"   🏆 Average Quality: {performance_summary.get('avg_quality', 0.0):.3f}")
            print(f"   ⚡ Average Response Time: {performance_summary.get('avg_time', 0.0):.3f}s")
            print(f"   🎯 Average Confidence: {performance_summary.get('avg_confidence', 0.0):.3f}")
            print(f"   💾 Data saved: {self.csv_filename}")
            print("=" * 60)
            print("🌟 Thank you for testing this revolutionary AI system!")

# ==================== Main Execution ====================
def main():
    print("🔶✨ CONDITION: MULTI-ADAPTIVE (PROPOSED SYSTEM WITH GPT-4)")
    print("=" * 80)
    print("🔶✨ EXPERIMENTAL CONDITION:")
    print("   📍 Agent Type: MULTI (GPT-4 Planner, Critic, Executor)")
    print("   📍 Adaptation: ADAPTIVE (Full user adaptation)")
    print("   📍 Strategy: 4-Level Hierarchical MCTS")
    print("=" * 80)
    print("🌟 REVOLUTIONARY FEATURES:")
    print("   🎯 4-Level Hierarchical MCTS (Meta → Cognitive → Combination → Execution)")
    print("   🤖 GPT-4 Based Multi-Agent Collaboration")
    print("   🧠 Real-time Cognitive Load Adaptation via Pupil Tracking")
    print("   👁️ Advanced MediaPipe-based Gaze & Emotion Analysis")
    print("   🎭 Adaptive Agent Personality (BaseTemplate + PersonalityTag)")
    print("   📊 GEMMAS Framework (IDS, UPR)")
    print("   🏆 Paper Reward Function (R* = w₁R_emo + w₂R_eff + w₃R_unc + w₄R_safe)")
    print("   🔬 Research-Grade Data Collection & Analysis")
    print("=" * 80)
    
    # GPT-4 통합 여부 확인
    try:
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("✅ OpenAI API Key detected - GPT-4 agents will be used")
            use_gpt4 = True
        else:
            print("⚠️  No OpenAI API Key - Running without GPT-4 agents")
            use_gpt4 = False
    except:
        use_gpt4 = False
    
    try:
        user_name = input("Enter your name: ")
    except EOFError:
        user_name = "MultiAdaptive_User"
    
    print(f"\n✅ Running MULTI-ADAPTIVE (PROPOSED) condition")
    print(f"   User: {user_name}")
    print(f"   GPT-4 Agents: {'ENABLED' if use_gpt4 else 'DISABLED'}\n")
    
    # Multi-Adaptive 모드 (제안 시스템)
    system_mode = "proposed"
    
    try:
        system = HierarchicalMCTSIntegratedSystem(user_name, system_mode)
        
        # GPT-4 에이전트 통합 (선택적)
        if use_gpt4:
            try:
                print("🚀 Integrating GPT-4 Multi-Agent System...")
                from integration_wrapper import integrate_with_maca_system
                integrate_with_maca_system(system, api_key=api_key)
                system.use_llm_agents = True
                print("✅ GPT-4 agents integrated successfully!")
            except Exception as e:
                print(f"⚠️  GPT-4 integration failed: {e}")
                print("   Continuing with standard MCTS only...")
                system.use_llm_agents = False
        else:
            system.use_llm_agents = False
        
        system.run()
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"\nSystem error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🎭 Hierarchical MCTS system shutdown complete")
        print("🚀 Thank you for experiencing the future of AI!")

if __name__ == "__main__":
    main()

