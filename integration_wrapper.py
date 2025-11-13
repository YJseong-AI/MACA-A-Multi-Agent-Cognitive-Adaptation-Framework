"""
통합 래퍼: 모든 구현을 maca.py와 통합

논문 요구사항에 따라 구현된 모든 컴포넌트를 통합:
1. GPT-4 기반 에이전트 (Planner, Critic, Executor)
2. 순차적 협력 프로토콜
3. 적응형 에이전트 성격
4. 논문 보상 함수
5. GEMMAS 프레임워크
"""

from typing import Dict, Any, Optional
import numpy as np

from llm_agents import PlannerAgent, CriticAgent, ExecutorAgent
from sequential_collaboration import SequentialCollaborationProtocol
from gemmas_framework import GEMMASFramework
from paper_reward_function import PaperRewardFunction


class IntegratedMultiAgentSystem:
    """
    통합 멀티 에이전트 시스템
    
    maca.py의 HierarchicalMCTSSystem에 통합되어 사용
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # 순차적 협력 프로토콜 (Planner, Critic, Executor 포함)
        self.collaboration_protocol = SequentialCollaborationProtocol(api_key)
        
        # GEMMAS 프레임워크
        self.gemmas = GEMMASFramework()
        
        # 논문 보상 함수
        self.reward_function = PaperRewardFunction()
        
        # 성능 추적
        self.collaboration_history = []
        self.ids_history = []
        self.upr_history = []
        
        print("✅ Integrated Multi-Agent System initialized")
    
    def process_decision(self, user_input: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        의사결정 처리: 순차적 협력 + GEMMAS 평가 + 보상 계산
        
        Args:
            user_input: 사용자 입력
            user_context: 사용자 컨텍스트 (감정, 인지 부하 등)
            
        Returns:
            Dict: 통합 의사결정 결과
        """
        # 1. 순차적 협력 처리 (Planner → Critic → Executor)
        collab_result = self.collaboration_protocol.process(user_input, user_context)
        
        # 2. GEMMAS로 협력 품질 평가
        ids, upr = self.gemmas.evaluate_collaboration_quality(collab_result.agent_messages)
        
        # 3. 논문 보상 함수로 보상 계산
        decision = {
            'meta_strategy': collab_result.strategy,
            'selected_action': collab_result.selected_action,
            'q_value': collab_result.q_value,
            'decision_time': collab_result.total_time,
            'tree_depth': 3,  # Planner → Critic → Executor (3 레벨)
            'confidence': self._calculate_confidence(collab_result, ids, upr)
        }
        
        total_reward = self.reward_function.calculate_total_reward(user_context, decision)
        
        # 4. 히스토리 저장
        self.collaboration_history.append(collab_result)
        self.ids_history.append(ids)
        self.upr_history.append(upr)
        
        # 5. 통합 결과 반환
        return {
            'strategy': collab_result.strategy,
            'selected_action': collab_result.selected_action,
            'execution_result': collab_result.execution_result,
            'q_value': collab_result.q_value,
            'total_reward': total_reward,
            'ids': ids,
            'upr': upr,
            'collaboration_quality': self.gemmas.get_collaboration_feedback(ids, upr),
            'decision_time': collab_result.total_time,
            'confidence': decision['confidence'],
            'agent_messages': collab_result.agent_messages
        }
    
    def _calculate_confidence(self, collab_result, ids: float, upr: float) -> float:
        """신뢰도 계산"""
        # Q-value 기반 신뢰도
        q_confidence = collab_result.q_value
        
        # GEMMAS 기반 신뢰도 (높은 IDS, 낮은 UPR → 높은 신뢰도)
        gemmas_confidence = ids * (1.0 - upr)
        
        # 가중 평균
        confidence = 0.6 * q_confidence + 0.4 * gemmas_confidence
        
        return np.clip(confidence, 0.0, 1.0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약"""
        if not self.collaboration_history:
            return {}
        
        return {
            'total_decisions': len(self.collaboration_history),
            'avg_ids': np.mean(self.ids_history) if self.ids_history else 0.0,
            'avg_upr': np.mean(self.upr_history) if self.upr_history else 0.0,
            'avg_decision_time': np.mean([c.total_time for c in self.collaboration_history]),
            'efficient_collaborations': sum(1 for i, u in zip(self.ids_history, self.upr_history) 
                                           if self.gemmas.is_efficient_collaboration(i, u))
        }


def integrate_with_maca_system(hierarchical_mcts_system, api_key: Optional[str] = None):
    """
    maca.py의 HierarchicalMCTSSystem에 통합
    
    Args:
        hierarchical_mcts_system: HierarchicalMCTSSystem 인스턴스
        api_key: OpenAI API 키 (선택사항)
    """
    # 통합 멀티 에이전트 시스템 추가
    hierarchical_mcts_system.integrated_agents = IntegratedMultiAgentSystem(api_key)
    
    print("✅ Integrated Multi-Agent System attached to HierarchicalMCTSSystem")
    
    # 기존 의사결정 메서드를 래핑하는 새 메서드 추가
    original_decision_making = hierarchical_mcts_system.hierarchical_decision_making
    
    def enhanced_decision_making(user_context: Dict, face_detected: bool):
        """향상된 의사결정 (GPT-4 에이전트 통합)"""
        # 기존 MCTS 의사결정
        mcts_decision = original_decision_making(user_context, face_detected)
        
        # GPT-4 에이전트 협력 (선택적)
        if hasattr(hierarchical_mcts_system, 'use_llm_agents') and hierarchical_mcts_system.use_llm_agents:
            try:
                agent_result = hierarchical_mcts_system.integrated_agents.process_decision(
                    user_input="Process MCTS decision",
                    user_context=user_context
                )
                
                # GEMMAS 품질 정보 추가
                mcts_decision.ids = agent_result['ids']
                mcts_decision.upr = agent_result['upr']
                mcts_decision.agent_feedback = agent_result['collaboration_quality']
            except Exception as e:
                print(f"⚠️ LLM Agent 처리 에러: {e}")
        
        return mcts_decision
    
    hierarchical_mcts_system.enhanced_decision_making = enhanced_decision_making
    
    return hierarchical_mcts_system

