"""
논문 요구사항에 따른 보상 함수 구현

R* = w1*R_emo + w2*R_eff + w3*R_unc + w4*R_safe

- R_emo: 감정 안정성 (ResEmoteNet 출력 사용)
- R_eff: 효율성 (시선 안정성 기반)
- R_unc: 불확실성 (일관성 없는 의사결정 벌칙)
- R_safe: 안전성 (과도한 탐색 제한)
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque


class PaperRewardFunction:
    """논문 요구사항에 따른 보상 함수"""
    
    def __init__(self):
        # 진화 알고리즘으로 최적화된 가중치 (초기값)
        # 3세대 진화 탐색으로 최적화됨 (population=10, mutation=0.1, crossover=0.5, threshold=0.005)
        self.weights = np.array([0.3, 0.3, 0.2, 0.2])  # [w1, w2, w3, w4]
        
        # 의사결정 히스토리 (불확실성 계산용)
        self.decision_history = deque(maxlen=50)
        
    def calculate_total_reward(self, user_context: Dict[str, Any], 
                              decision: Dict[str, Any]) -> float:
        """
        총 보상 계산: R* = w1*R_emo + w2*R_eff + w3*R_unc + w4*R_safe
        
        Args:
            user_context: 사용자 컨텍스트 (감정, 시선, 인지 부하 등)
            decision: 의사결정 정보
            
        Returns:
            float: 총 보상 값 (0.0-1.0)
        """
        # R_emo: 감정 안정성 (ResEmoteNet 출력 사용)
        R_emo = self.calculate_emotion_stability_reward(user_context)
        
        # R_eff: 효율성 (시선 안정성 기반)
        R_eff = self.calculate_efficiency_reward(user_context)
        
        # R_unc: 불확실성 (일관성 없는 의사결정 벌칙)
        R_unc = self.calculate_uncertainty_reward(user_context, decision)
        
        # R_safe: 안전성 (과도한 탐색 제한)
        R_safe = self.calculate_safety_reward(decision)
        
        # 가중 합계
        R_star = (self.weights[0] * R_emo + 
                 self.weights[1] * R_eff + 
                 self.weights[2] * R_unc + 
                 self.weights[3] * R_safe)
        
        # 의사결정 히스토리 업데이트
        self.decision_history.append(decision)
        
        return np.clip(R_star, 0.0, 1.0)
    
    def calculate_emotion_stability_reward(self, user_context: Dict[str, Any]) -> float:
        """
        R_emo: 감정 안정성 보상
        
        ResEmoteNet 출력을 직접 사용하여 감정 안정성 계산
        논문: "ResEmoteNet outputs probabilistic distributions across seven basic emotions,
        which are used to compute the emotional stability component (R_emo) of the reward function."
        
        Args:
            user_context: 사용자 컨텍스트
            
        Returns:
            float: 감정 안정성 보상 (0.0-1.0)
        """
        # ResEmoteNet 출력 (7가지 감정에 대한 확률 분포)
        emotion_probs = user_context.get('emotion_probabilities', {})
        
        if not emotion_probs:
            # 기본값: 중립 감정
            return 0.5
        
        # 감정 안정성 = 최대 확률 (명확한 감정일수록 안정적)
        max_prob = max(emotion_probs.values()) if emotion_probs else 0.0
        
        # 부정적 감정(sad, fear, anger)에 대한 페널티
        negative_emotions = ['sad', 'fear', 'anger', 'disgust']
        negative_prob = sum(emotion_probs.get(emotion, 0.0) for emotion in negative_emotions)
        
        # 안정성 점수 = 최대 확률 - 부정적 감정 페널티
        stability_score = max_prob - 0.3 * negative_prob
        
        return np.clip(stability_score, 0.0, 1.0)
    
    def calculate_efficiency_reward(self, user_context: Dict[str, Any]) -> float:
        """
        R_eff: 효율성 보상
        
        시선 안정성 기반 효율성 계산
        논문: "These values serve as key inputs at Level 2 of the system to evaluate 
        efficiency (R_eff) and sustained attention."
        
        Args:
            user_context: 사용자 컨텍스트
            
        Returns:
            float: 효율성 보상 (0.0-1.0)
        """
        # 시선 안정성 (Fixation Stability)
        fixation_stability = user_context.get('fixation_stability', 0.5)
        
        # 주의 집중도
        attention = user_context.get('attention', 0.5)
        
        # 동공 크기 (인지 부하 지표)
        pupil_size = user_context.get('pupil_size', 0.5)
        
        # 효율성 = 시선 안정성 * 주의 집중도 * (1 - 정규화된 동공 크기)
        # 동공이 너무 크면 인지 부하가 높아 효율성 감소
        normalized_pupil = np.clip(pupil_size, 0.0, 1.0)
        efficiency = fixation_stability * attention * (1.0 - normalized_pupil * 0.5)
        
        return np.clip(efficiency, 0.0, 1.0)
    
    def calculate_uncertainty_reward(self, user_context: Dict[str, Any], 
                                    decision: Dict[str, Any]) -> float:
        """
        R_unc: 불확실성 보상 (일관성 없는 의사결정 벌칙)
        
        논문: "The uncertainty (R_unc) term penalized inconsistent decision traces"
        
        Args:
            user_context: 사용자 컨텍스트
            decision: 현재 의사결정
            
        Returns:
            float: 불확실성 보상 (0.0-1.0, 높을수록 일관적)
        """
        if len(self.decision_history) < 2:
            # 히스토리가 부족하면 중간값 반환
            return 0.5
        
        # 최근 의사결정들의 일관성 측정
        recent_decisions = list(self.decision_history)[-10:]  # 최근 10개
        
        # 의사결정 추적의 일관성 계산
        consistency_score = self._calculate_decision_consistency(recent_decisions, decision)
        
        # 불확실성 = 1 - 일관성 (일관적일수록 불확실성 낮음)
        uncertainty = 1.0 - consistency_score
        
        # 보상은 일관성에 비례 (일관적일수록 높은 보상)
        return np.clip(consistency_score, 0.0, 1.0)
    
    def _calculate_decision_consistency(self, recent_decisions: List[Dict], 
                                       current_decision: Dict) -> float:
        """의사결정 일관성 계산"""
        if not recent_decisions:
            return 0.5
        
        # 메타 전략 일관성
        meta_strategies = [d.get('meta_strategy', '') for d in recent_decisions]
        current_meta = current_decision.get('meta_strategy', '')
        
        # 최근 의사결정과 현재 의사결정의 유사도
        meta_consistency = sum(1 for m in meta_strategies if m == current_meta) / len(meta_strategies)
        
        # 인지 적응 일관성
        adaptations = [d.get('cognitive_adaptation', '') for d in recent_decisions]
        current_adaptation = current_decision.get('cognitive_adaptation', '')
        adaptation_consistency = sum(1 for a in adaptations if a == current_adaptation) / len(adaptations)
        
        # 평균 일관성
        consistency = (meta_consistency + adaptation_consistency) / 2.0
        
        return np.clip(consistency, 0.0, 1.0)
    
    def calculate_safety_reward(self, decision: Dict[str, Any]) -> float:
        """
        R_safe: 안전성 보상 (과도한 탐색 제한)
        
        논문: "The safety (R_safe) term constrained excessive exploration to maintain 
        stable reasoning paths."
        
        Args:
            decision: 의사결정 정보
            
        Returns:
            float: 안전성 보상 (0.0-1.0)
        """
        # 의사결정 시간 (너무 오래 걸리면 안전하지 않음)
        decision_time = decision.get('decision_time', 0.0)
        
        # 탐색 깊이 (너무 깊으면 안전하지 않음)
        tree_depth = decision.get('tree_depth', 0)
        
        # 신뢰도 (높을수록 안전)
        confidence = decision.get('confidence', 0.5)
        
        # 안전성 점수 계산
        # 시간이 적당하고, 깊이가 적당하며, 신뢰도가 높을수록 안전
        time_safety = 1.0 - np.clip(decision_time / 5.0, 0.0, 1.0)  # 5초 이상이면 페널티
        depth_safety = 1.0 - np.clip((tree_depth - 3) / 5.0, 0.0, 1.0)  # 깊이 3이 적정
        confidence_safety = confidence
        
        # 평균 안전성
        safety = (time_safety + depth_safety + confidence_safety) / 3.0
        
        return np.clip(safety, 0.0, 1.0)
    
    def evolve_weights(self, population_size=10, generations=3, 
                        mutation_rate=0.1, crossover_rate=0.5, 
                        convergence_threshold=0.005):
        """
        가중치 진화 최적화
        
        논문: "Weights w were optimized through a three-generation evolutionary search
        (population = 10, mutation = 0.1, crossover = 0.5, threshold = 0.005)"
        
        Args:
            population_size: 개체군 크기
            generations: 세대 수
            mutation_rate: 변이율
            crossover_rate: 교배율
            convergence_threshold: 수렴 임계값
        """
        # 초기 개체군 생성 (가중치 벡터)
        population = []
        for _ in range(population_size):
            weights = np.random.dirichlet([1, 1, 1, 1])  # 정규화된 가중치
            population.append(weights)
        
        # 진화 반복
        for generation in range(generations):
            # 적합도 평가 (간단한 휴리스틱 사용)
            fitness_scores = [self._evaluate_weight_fitness(w) for w in population]
            
            # 수렴 확인
            if max(fitness_scores) - min(fitness_scores) < convergence_threshold:
                break
            
            # 선택, 교배, 변이
            new_population = []
            
            # 엘리트 보존 (상위 20%)
            elite_size = max(1, population_size // 5)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # 나머지는 교배와 변이로 생성
            while len(new_population) < population_size:
                # 토너먼트 선택
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # 교배
                child = self._crossover(parent1, parent2, crossover_rate)
                
                # 변이
                child = self._mutate(child, mutation_rate)
                
                new_population.append(child)
            
            population = new_population
        
        # 최적 가중치 선택
        final_fitness = [self._evaluate_weight_fitness(w) for w in population]
        best_idx = np.argmax(final_fitness)
        self.weights = population[best_idx]
        
        return self.weights
    
    def _evaluate_weight_fitness(self, weights: np.ndarray) -> float:
        """가중치 적합도 평가 (간단한 휴리스틱)"""
        # 가중치가 균형잡혀 있고, 합이 1에 가까울수록 좋음
        weight_sum = np.sum(weights)
        weight_balance = 1.0 - np.std(weights)  # 표준편차가 작을수록 균형
        
        fitness = 0.5 * (1.0 - abs(weight_sum - 1.0)) + 0.5 * weight_balance
        return fitness
    
    def _tournament_selection(self, population: List[np.ndarray], 
                             fitness_scores: List[float], tournament_size=2) -> np.ndarray:
        """토너먼트 선택"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, 
                  crossover_rate: float) -> np.ndarray:
        """교배"""
        if np.random.random() < crossover_rate:
            # 가중 평균
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            # 정규화
            child = child / np.sum(child)
        else:
            child = parent1.copy()
        return child
    
    def _mutate(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
        """변이"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # 가우시안 노이즈 추가
                noise = np.random.normal(0, 0.05)
                mutated[i] = np.clip(mutated[i] + noise, 0.01, 0.99)
        
        # 정규화
        mutated = mutated / np.sum(mutated)
        return mutated

