"""
순차적 협력 프로토콜 구현

논문 요구사항:
"User input and emotional context are first passed to the Planner, 
followed by sequential processing through the Critic and Executor."

Planner → Critic → Executor 순차 처리
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from llm_agents import (
    PlannerAgent, CriticAgent, ExecutorAgent,
    SharedContextBuffer, AdaptiveAgentPersonality
)


@dataclass
class CollaborationResult:
    """순차적 협력 결과"""
    strategy: str
    selected_action: str
    execution_result: Dict[str, Any]
    q_value: float
    total_time: float
    agent_messages: List[Dict[str, Any]]


class SequentialCollaborationProtocol:
    """
    순차적 협력 프로토콜
    
    논문: "User input and emotional context are first passed to the Planner, 
          followed by sequential processing through the Critic and Executor"
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # 3개의 에이전트 초기화
        self.planner = PlannerAgent(api_key)
        self.critic = CriticAgent(api_key)
        self.executor = ExecutorAgent(api_key)
        
        # 공유 컨텍스트 버퍼
        self.context_buffer = SharedContextBuffer(max_size=50)
        
        # 적응형 성격 시스템
        self.personality_system = AdaptiveAgentPersonality()
        
        print("✅ Sequential Collaboration Protocol initialized")
        print(f"   - Planner: {'Real GPT-4' if not self.planner.simulation_mode else 'Simulation'}")
        print(f"   - Critic: {'Real GPT-4' if not self.critic.simulation_mode else 'Simulation'}")
        print(f"   - Executor: {'Real GPT-4' if not self.executor.simulation_mode else 'Simulation'}")
    
    def process(self, user_input: str, user_context: Dict[str, Any]) -> CollaborationResult:
        """
        순차적 협력 처리: Planner → Critic → Executor
        
        Args:
            user_input: 사용자 입력
            user_context: 사용자 컨텍스트 (감정, 인지 부하 등)
            
        Returns:
            CollaborationResult: 협력 결과
        """
        start_time = time.time()
        agent_messages = []
        
        # 사용자 컨텍스트 업데이트
        self.context_buffer.update_user_context(user_context)
        
        # 감정 안정성(E_t)에 따른 성격 프로필 결정
        emotion = user_context.get('emotion', 'neutral')
        
        # === Step 1: Planner - 고수준 전략 생성 ===
        planner_personality = self.personality_system.generate_personality_tag('planner', emotion)
        strategy_data = self.planner.generate_strategy(self.context_buffer, planner_personality)
        
        agent_messages.append({
            "agent": "planner",
            "personality": planner_personality,
            "output": strategy_data
        })
        
        # === Step 2: Critic - 전략 평가 및 행동 선택 ===
        critic_personality = self.personality_system.generate_personality_tag('critic', emotion)
        
        # 후보 행동 생성 (전략 기반)
        candidate_actions = self._generate_candidate_actions(strategy_data, user_context)
        
        # 각 반복에서 Critic은 후보 행동(Q_i) 평가 및 최적 행동(a*) 선택
        evaluation_data = self.critic.evaluate_actions(
            self.context_buffer, 
            critic_personality,
            candidate_actions
        )
        
        agent_messages.append({
            "agent": "critic",
            "personality": critic_personality,
            "output": evaluation_data
        })
        
        # 최적 행동 선택
        selected_action = evaluation_data.get('best_action', evaluation_data.get('selected_action', 'default'))
        q_value = evaluation_data.get('calculated_q_values', {}).get(selected_action, 0.5)
        
        # === Step 3: Executor - 행동 실행 ===
        executor_personality = self.personality_system.generate_personality_tag('executor', emotion)
        
        action_to_execute = {
            "action": selected_action,
            "strategy": strategy_data.get('strategy', 'adaptive'),
            "user_input": user_input
        }
        
        execution_result = self.executor.execute_action(
            self.context_buffer,
            executor_personality,
            action_to_execute
        )
        
        agent_messages.append({
            "agent": "executor",
            "personality": executor_personality,
            "output": execution_result
        })
        
        # 총 처리 시간
        total_time = time.time() - start_time
        
        return CollaborationResult(
            strategy=strategy_data.get('strategy', 'adaptive'),
            selected_action=selected_action,
            execution_result=execution_result,
            q_value=q_value,
            total_time=total_time,
            agent_messages=agent_messages
        )
    
    def _generate_candidate_actions(self, strategy_data: Dict[str, Any], 
                                   user_context: Dict[str, Any]) -> List[str]:
        """전략과 컨텍스트 기반 후보 행동 생성"""
        # strategy_data가 dict이므로 strategy 키 추출
        if isinstance(strategy_data, dict):
            strategy = strategy_data.get('strategy', 'adaptive')
        else:
            strategy = str(strategy_data)
        
        # strategy가 여전히 dict면 문자열로 변환
        if isinstance(strategy, dict):
            strategy = 'adaptive'  # 기본값
        
        emotion = user_context.get('emotion', 'neutral')
        cognitive_load = user_context.get('cognitive_load_level', 'medium')
        
        # 전략별 기본 행동들
        strategy_actions = {
            'combination': ['combine_agents', 'optimize_combination', 'parallel_execution'],
            'collaboration': ['sequential_collaboration', 'agent_communication', 'shared_reasoning'],
            'hybrid': ['mixed_strategy', 'adaptive_switching', 'dynamic_allocation'],
            'adaptive': ['context_adaptation', 'personalized_approach', 'flexible_execution'],
            'dynamic': ['real_time_adjustment', 'continuous_optimization', 'proactive_planning']
        }
        
        # 감정 기반 행동 수정
        if emotion in ['sad', 'fear', 'anger']:
            # 부정적 감정 → 지지적 행동 추가
            supportive_actions = ['provide_support', 'gentle_guidance', 'reassurance']
            base_actions = strategy_actions.get(strategy, ['default_action'])
            return base_actions[:2] + supportive_actions[:1]
        
        elif emotion in ['happy', 'surprise']:
            # 긍정적 감정 → 도전적 행동 추가
            challenging_actions = ['challenging_task', 'intensive_processing', 'advanced_strategy']
            base_actions = strategy_actions.get(strategy, ['default_action'])
            return base_actions[:2] + challenging_actions[:1]
        
        # 기본 행동 반환
        return strategy_actions.get(strategy, ['default_action', 'standard_processing', 'balanced_approach'])
    
    def get_context_buffer(self) -> SharedContextBuffer:
        """공유 컨텍스트 버퍼 반환"""
        return self.context_buffer
    
    def reset_context(self):
        """컨텍스트 초기화"""
        self.context_buffer = SharedContextBuffer(max_size=50)

