"""
GPT-4 기반 언어 모델 에이전트 구현

논문 요구사항:
- Planner: 고수준 전략 생성
- Critic: 예상 보상(Q) 추정하여 전략 평가
- Executor: 선택된 행동 수행 및 MCTS 롤아웃 결과 반환
- 각 에이전트는 공통 GPT-4 추론 코어 공유
- 공유 컨텍스트 버퍼를 통해 정보 교환 (JSON 형식)
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI 라이브러리 없음 - 시뮬레이션 모드로 실행")


@dataclass
class AgentMessage:
    """에이전트 간 메시지 구조 (JSON 형식)"""
    agent_type: str  # "planner", "critic", "executor"
    timestamp: float
    content: Dict[str, Any]
    personality: str
    
    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps({
            "agent_type": self.agent_type,
            "timestamp": self.timestamp,
            "content": self.content,
            "personality": self.personality
        }, indent=2)


class SharedContextBuffer:
    """공유 컨텍스트 버퍼 - 에이전트 간 정보 교환"""
    
    def __init__(self, max_size=50):
        self.messages = deque(maxlen=max_size)
        self.user_context = {}
        
    def add_message(self, message: AgentMessage):
        """메시지 추가"""
        self.messages.append(message)
        
    def get_recent_messages(self, n=10) -> List[AgentMessage]:
        """최근 n개 메시지 가져오기"""
        return list(self.messages)[-n:]
    
    def update_user_context(self, context: Dict[str, Any]):
        """사용자 컨텍스트 업데이트"""
        self.user_context.update(context)
    
    def get_context_summary(self) -> str:
        """컨텍스트 요약 (GPT-4 프롬프트용)"""
        recent_msgs = self.get_recent_messages(5)
        
        summary = f"User Context:\n"
        summary += f"- Emotion: {self.user_context.get('emotion', 'neutral')}\n"
        summary += f"- Attention: {self.user_context.get('attention', 0.5):.2f}\n"
        summary += f"- Cognitive Load: {self.user_context.get('cognitive_load_level', 'medium')}\n\n"
        
        summary += "Recent Agent Interactions:\n"
        for msg in recent_msgs:
            summary += f"- [{msg.agent_type}] {json.dumps(msg.content)}\n"
        
        return summary


class AdaptiveAgentPersonality:
    """
    적응형 에이전트 성격 시스템
    
    논문: "각 에이전트는 사용자의 감정 상태에 적응하는 동적 성격 프로필을 유지합니다"
    Prompt_agent = BaseTemplate + PersonalityTag(E_t)
    """
    
    def __init__(self):
        self.personality_profiles = {
            'planner': {
                'encouraging': "You are an encouraging planner who provides supportive and motivating strategies.",
                'calming': "You are a calming planner who provides gentle and reassuring strategies.",
                'challenging': "You are a challenging planner who provides ambitious and demanding strategies.",
                'motivating': "You are a motivating planner who provides energizing and inspiring strategies.",
                'adaptive': "You are an adaptive planner who provides balanced and flexible strategies."
            },
            'critic': {
                'supportive': "You are a supportive critic who provides constructive and empathetic feedback.",
                'understanding': "You are an understanding critic who provides patient and compassionate evaluations.",
                'thorough': "You are a thorough critic who provides detailed and rigorous analysis.",
                'optimistic': "You are an optimistic critic who focuses on positive aspects and potential.",
                'balanced': "You are a balanced critic who provides fair and objective assessments."
            },
            'executor': {
                'gentle': "You are a gentle executor who implements actions carefully and kindly.",
                'careful': "You are a careful executor who proceeds with caution and precision.",
                'intensive': "You are an intensive executor who performs actions with high energy and focus.",
                'energetic': "You are an energetic executor who acts with enthusiasm and vigor.",
                'standard': "You are a standard executor who performs actions efficiently and reliably."
            }
        }
    
    def generate_personality_tag(self, agent_type: str, emotion: str) -> str:
        """
        PersonalityTag(E_t) 생성
        
        논문: "부정적 감정(sad, fear, anger) → encouraging/supportive/gentle
              긍정적 감정(happy, neutral) → challenging/intensive"
        """
        if emotion in ['sad', 'fear', 'anger']:
            # 부정적 감정 → 지지적 성격
            personality_map = {
                'planner': 'encouraging',
                'critic': 'supportive',
                'executor': 'gentle'
            }
        elif emotion in ['happy', 'surprise']:
            # 긍정적 감정 → 도전적 성격
            personality_map = {
                'planner': 'challenging',
                'critic': 'thorough',
                'executor': 'intensive'
            }
        else:  # neutral
            # 중립 → 균형잡힌 성격
            personality_map = {
                'planner': 'adaptive',
                'critic': 'balanced',
                'executor': 'standard'
            }
        
        personality = personality_map.get(agent_type, 'adaptive')
        return personality
    
    def get_personality_prompt(self, agent_type: str, personality: str) -> str:
        """성격 프로필에 맞는 프롬프트 가져오기"""
        return self.personality_profiles.get(agent_type, {}).get(personality, "")


class GPTAgentBase:
    """GPT-4 기반 에이전트 베이스 클래스"""
    
    def __init__(self, agent_type: str, api_key: Optional[str] = None):
        self.agent_type = agent_type
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.personality_system = AdaptiveAgentPersonality()
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.simulation_mode = False
        else:
            self.client = None
            self.simulation_mode = True
            print(f"⚠️ {agent_type} - 시뮬레이션 모드로 실행")
        
        # 에이전트별 BaseTemplate
        self.base_templates = {
            'planner': """You are a high-level strategic planner in a multi-agent decision system.
Your role is to generate high-level strategies based on user context.
Provide your response in JSON format with 'strategy' and 'reasoning' fields.""",
            
            'critic': """You are a critical evaluator in a multi-agent decision system.
Your role is to evaluate strategies by estimating expected rewards (Q-values).
Provide your response in JSON format with 'q_value', 'evaluation', and 'selected_action' fields.""",
            
            'executor': """You are an action executor in a multi-agent decision system.
Your role is to perform selected actions and return execution results for MCTS rollouts.
Provide your response in JSON format with 'execution_result', 'outcome', and 'feedback' fields."""
        }
    
    def _call_gpt4(self, prompt: str, temperature=0.7) -> str:
        """GPT-4 API 호출"""
        if self.simulation_mode or not self.client:
            # 시뮬레이션 모드: 더미 응답 반환
            return self._generate_dummy_response()
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.base_templates[self.agent_type]},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ GPT-4 API 에러: {e}")
            return self._generate_dummy_response()
    
    def _generate_dummy_response(self) -> str:
        """시뮬레이션 모드용 더미 응답"""
        if self.agent_type == 'planner':
            return json.dumps({
                "strategy": "combination",
                "reasoning": "Based on user context, combination strategy is optimal."
            })
        elif self.agent_type == 'critic':
            return json.dumps({
                "q_value": 0.75,
                "evaluation": "Strategy has high expected reward.",
                "selected_action": "proceed"
            })
        else:  # executor
            return json.dumps({
                "execution_result": "success",
                "outcome": "Action completed successfully.",
                "feedback": "User response positive."
            })


class PlannerAgent(GPTAgentBase):
    """
    Planner 에이전트: 고수준 전략 생성
    
    논문: "The Planner generates high-level strategies"
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("planner", api_key)
    
    def generate_strategy(self, context_buffer: SharedContextBuffer, 
                         current_personality: str) -> Dict[str, Any]:
        """
        고수준 전략 생성
        
        Args:
            context_buffer: 공유 컨텍스트 버퍼
            current_personality: 현재 성격 프로필
            
        Returns:
            Dict: 전략 정보
        """
        # BaseTemplate + PersonalityTag 결합
        personality_prompt = self.personality_system.get_personality_prompt('planner', current_personality)
        context_summary = context_buffer.get_context_summary()
        
        full_prompt = f"""{personality_prompt}

{context_summary}

Generate a high-level strategy for the current situation. Consider:
1. User's emotional state
2. Cognitive load level
3. Attention capacity

Provide your response in JSON format with 'strategy' and 'reasoning' fields."""
        
        # GPT-4 호출
        response = self._call_gpt4(full_prompt)
        
        try:
            strategy_data = json.loads(response)
        except:
            strategy_data = {"strategy": "adaptive", "reasoning": "Default strategy"}
        
        # 메시지 추가
        message = AgentMessage(
            agent_type="planner",
            timestamp=time.time(),
            content=strategy_data,
            personality=current_personality
        )
        context_buffer.add_message(message)
        
        return strategy_data


class CriticAgent(GPTAgentBase):
    """
    Critic 에이전트: 예상 보상(Q) 추정하여 전략 평가
    
    논문: "The Critic evaluates them by estimating expected rewards (Q)"
    """
    
    def __init__(self, api_key: Optional[str] = None, gamma=0.9):
        super().__init__("critic", api_key)
        self.gamma = gamma  # 할인 계수
        self.value_estimates = {}  # V(s) 추정값 저장
    
    def evaluate_actions(self, context_buffer: SharedContextBuffer,
                        current_personality: str,
                        candidate_actions: List[str]) -> Dict[str, Any]:
        """
        후보 행동들의 Q-value 평가
        
        논문: "At each iteration, the Critic evaluates candidate actions (Q_i) 
              and selects the optimal one (a*) for execution."
        
        Q(s_t, a_t) = R_t + γV(s_{t+1})
        """
        # BaseTemplate + PersonalityTag 결합
        personality_prompt = self.personality_system.get_personality_prompt('critic', current_personality)
        context_summary = context_buffer.get_context_summary()
        
        full_prompt = f"""{personality_prompt}

{context_summary}

Evaluate the following candidate actions and estimate their Q-values:
{json.dumps(candidate_actions, indent=2)}

For each action, calculate Q(s_t, a_t) = R_t + γV(s_{{t+1}}) where:
- R_t is the immediate reward
- γ is the discount factor ({self.gamma})
- V(s_{{t+1}}) is the estimated value of the next state

Provide your response in JSON format with 'q_values', 'selected_action', and 'evaluation' fields."""
        
        # GPT-4 호출
        response = self._call_gpt4(full_prompt)
        
        try:
            evaluation_data = json.loads(response)
        except:
            evaluation_data = {
                "q_values": {action: 0.5 for action in candidate_actions},
                "selected_action": candidate_actions[0] if candidate_actions else "default",
                "evaluation": "Default evaluation"
            }
        
        # Q-value 계산 (논문 공식 적용)
        q_values = self._calculate_q_values(context_buffer, candidate_actions)
        evaluation_data['calculated_q_values'] = q_values
        evaluation_data['best_action'] = max(q_values, key=q_values.get)
        
        # 메시지 추가
        message = AgentMessage(
            agent_type="critic",
            timestamp=time.time(),
            content=evaluation_data,
            personality=current_personality
        )
        context_buffer.add_message(message)
        
        return evaluation_data
    
    def _calculate_q_values(self, context_buffer: SharedContextBuffer,
                           actions: List[str]) -> Dict[str, float]:
        """
        Q-value 계산: Q(s_t, a_t) = R_t + γV(s_{t+1})
        
        논문 요구사항에 따른 Q-value 계산
        """
        q_values = {}
        current_state = self._get_state_representation(context_buffer)
        
        for action in actions:
            # R_t: 즉시 보상 추정
            immediate_reward = self._estimate_immediate_reward(context_buffer, action)
            
            # V(s_{t+1}): 다음 상태의 가치 추정
            next_state_value = self._estimate_next_state_value(context_buffer, action)
            
            # Q(s_t, a_t) = R_t + γV(s_{t+1})
            q_values[action] = immediate_reward + self.gamma * next_state_value
        
        return q_values
    
    def _get_state_representation(self, context_buffer: SharedContextBuffer) -> str:
        """현재 상태 표현"""
        ctx = context_buffer.user_context
        return f"{ctx.get('emotion', 'neutral')}_{ctx.get('cognitive_load_level', 'medium')}"
    
    def _estimate_immediate_reward(self, context_buffer: SharedContextBuffer, 
                                   action: str) -> float:
        """즉시 보상 R_t 추정"""
        # 간단한 휴리스틱 기반 보상 추정
        emotion = context_buffer.user_context.get('emotion', 'neutral')
        
        # 부정적 감정 시 지지적 행동에 높은 보상
        if emotion in ['sad', 'fear', 'anger']:
            supportive_actions = ['encouraging', 'supportive', 'gentle', 'collaboration']
            if any(keyword in action.lower() for keyword in supportive_actions):
                return 0.8
        
        # 긍정적 감정 시 도전적 행동에 높은 보상
        elif emotion in ['happy', 'surprise']:
            challenging_actions = ['challenging', 'intensive', 'hybrid']
            if any(keyword in action.lower() for keyword in challenging_actions):
                return 0.8
        
        return 0.5  # 기본 보상
    
    def _estimate_next_state_value(self, context_buffer: SharedContextBuffer,
                                   action: str) -> float:
        """다음 상태의 가치 V(s_{t+1}) 추정"""
        # 간단한 휴리스틱 기반 상태 가치 추정
        current_state = self._get_state_representation(context_buffer)
        
        if current_state in self.value_estimates:
            return self.value_estimates[current_state]
        
        # 기본 가치 추정
        return 0.6


class ExecutorAgent(GPTAgentBase):
    """
    Executor 에이전트: 선택된 행동 수행 및 MCTS 롤아웃 결과 반환
    
    논문: "The Executor performs the selected actions while returning outcomes 
          for MCTS rollouts"
    """
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("executor", api_key)
        self.execution_history = deque(maxlen=100)
    
    def execute_action(self, context_buffer: SharedContextBuffer,
                      current_personality: str,
                      action: Dict[str, Any]) -> Dict[str, Any]:
        """
        선택된 행동 실행 및 결과 반환
        
        Args:
            context_buffer: 공유 컨텍스트 버퍼
            current_personality: 현재 성격 프로필
            action: 실행할 행동
            
        Returns:
            Dict: 실행 결과 (MCTS 롤아웃용)
        """
        # BaseTemplate + PersonalityTag 결합
        personality_prompt = self.personality_system.get_personality_prompt('executor', current_personality)
        context_summary = context_buffer.get_context_summary()
        
        full_prompt = f"""{personality_prompt}

{context_summary}

Execute the following action:
{json.dumps(action, indent=2)}

Provide execution results for MCTS rollout evaluation. Include:
1. Execution success/failure
2. Observed outcome
3. User feedback simulation
4. Rollout reward estimate

Provide your response in JSON format with 'execution_result', 'outcome', 'reward', and 'feedback' fields."""
        
        # GPT-4 호출
        response = self._call_gpt4(full_prompt)
        
        try:
            execution_data = json.loads(response)
        except:
            execution_data = {
                "execution_result": "success",
                "outcome": "Action executed",
                "reward": 0.7,
                "feedback": "Positive"
            }
        
        # 실행 히스토리 저장
        self.execution_history.append({
            "action": action,
            "result": execution_data,
            "timestamp": time.time()
        })
        
        # 메시지 추가
        message = AgentMessage(
            agent_type="executor",
            timestamp=time.time(),
            content=execution_data,
            personality=current_personality
        )
        context_buffer.add_message(message)
        
        return execution_data
    
    def get_rollout_reward(self, context_buffer: SharedContextBuffer) -> float:
        """MCTS 롤아웃 보상 계산"""
        if not self.execution_history:
            return 0.5
        
        # 최근 실행 결과들의 평균 보상
        recent_executions = list(self.execution_history)[-5:]
        rewards = [exec_data['result'].get('reward', 0.5) for exec_data in recent_executions]
        
        return sum(rewards) / len(rewards)

