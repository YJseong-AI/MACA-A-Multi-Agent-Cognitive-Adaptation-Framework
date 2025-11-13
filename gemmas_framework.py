"""
GEMMAS 프레임워크 구현

논문 요구사항:
"To quantify cooperative quality, we applied the GEMMAS framework, 
which models inter-agent interactions as a directed acyclic graph.
Information Diversity Score (IDS) measures semantic complementarity, 
while Unnecessary Path Ratio (UPR) captures redundant reasoning."
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import json


class AgentInteractionGraph:
    """
    에이전트 간 상호작용을 방향성 비순환 그래프(DAG)로 모델링
    """
    
    def __init__(self):
        self.nodes = []  # 노드: (agent_type, message_content, timestamp)
        self.edges = []  # 간선: (from_idx, to_idx, interaction_type)
        self.adjacency_list = defaultdict(list)
        
    def add_node(self, agent_type: str, message_content: Dict[str, Any], 
                 timestamp: float) -> int:
        """노드 추가"""
        node_idx = len(self.nodes)
        self.nodes.append({
            'agent_type': agent_type,
            'content': message_content,
            'timestamp': timestamp,
            'idx': node_idx
        })
        return node_idx
    
    def add_edge(self, from_idx: int, to_idx: int, interaction_type: str = 'sequential'):
        """간선 추가 (방향성)"""
        self.edges.append({
            'from': from_idx,
            'to': to_idx,
            'type': interaction_type
        })
        self.adjacency_list[from_idx].append(to_idx)
    
    def is_dag(self) -> bool:
        """DAG 여부 확인 (순환 구조 검사)"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_idx):
            visited.add(node_idx)
            rec_stack.add(node_idx)
            
            for neighbor in self.adjacency_list[node_idx]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_idx)
            return False
        
        for node_idx in range(len(self.nodes)):
            if node_idx not in visited:
                if has_cycle(node_idx):
                    return False
        
        return True
    
    def get_paths(self) -> List[List[int]]:
        """모든 경로 찾기"""
        paths = []
        
        def dfs(node_idx, path):
            path.append(node_idx)
            
            if not self.adjacency_list[node_idx]:  # 리프 노드
                paths.append(path.copy())
            else:
                for neighbor in self.adjacency_list[node_idx]:
                    dfs(neighbor, path)
            
            path.pop()
        
        # 루트 노드들에서 시작
        root_nodes = [i for i in range(len(self.nodes)) 
                     if not any(i in self.adjacency_list[j] for j in range(len(self.nodes)))]
        
        for root in root_nodes:
            dfs(root, [])
        
        return paths


class GEMMASFramework:
    """
    GEMMAS: Graph-based Evaluation Metrics for Multi-Agent Systems
    
    논문 요구사항:
    - IDS (Information Diversity Score): 의미론적 보완성 측정
    - UPR (Unnecessary Path Ratio): 중복 추론 포착
    - 높은 IDS와 낮은 UPR → 효율적인 협력
    """
    
    def __init__(self):
        self.interaction_history = []
        
    def evaluate_collaboration_quality(self, agent_messages: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        협력 품질 평가
        
        Args:
            agent_messages: 에이전트 메시지 리스트
            
        Returns:
            Tuple[float, float]: (IDS, UPR)
        """
        # DAG 구성
        graph = self._build_interaction_graph(agent_messages)
        
        # IDS 계산: 정보 다양성 점수
        ids = self._calculate_ids(graph, agent_messages)
        
        # UPR 계산: 불필요한 경로 비율
        upr = self._calculate_upr(graph)
        
        return ids, upr
    
    def _build_interaction_graph(self, agent_messages: List[Dict[str, Any]]) -> AgentInteractionGraph:
        """에이전트 메시지로부터 상호작용 그래프 구성"""
        graph = AgentInteractionGraph()
        
        # 노드 추가 (순차적으로 Planner → Critic → Executor)
        for i, msg in enumerate(agent_messages):
            graph.add_node(
                agent_type=msg.get('agent', 'unknown'),
                message_content=msg.get('output', {}),
                timestamp=i
            )
        
        # 간선 추가 (순차적 상호작용)
        for i in range(len(agent_messages) - 1):
            graph.add_edge(i, i + 1, 'sequential')
        
        return graph
    
    def _calculate_ids(self, graph: AgentInteractionGraph, 
                      agent_messages: List[Dict[str, Any]]) -> float:
        """
        Information Diversity Score (IDS) 계산
        
        의미론적 보완성 측정: 각 에이전트가 얼마나 다양하고 보완적인 정보를 제공하는지
        
        논문: "Information Diversity Score (IDS) measures semantic complementarity"
        """
        if len(graph.nodes) < 2:
            return 0.0
        
        # 각 에이전트의 메시지 내용 추출
        message_contents = []
        for node in graph.nodes:
            content = node['content']
            # 메시지 내용을 문자열로 변환
            content_str = json.dumps(content, sort_keys=True)
            message_contents.append(content_str)
        
        # 의미론적 다양성 계산 (간단한 휴리스틱)
        diversity_scores = []
        
        for i in range(len(message_contents)):
            for j in range(i + 1, len(message_contents)):
                # 두 메시지 간 차이 계산 (Jaccard 유사도 기반)
                similarity = self._calculate_jaccard_similarity(
                    message_contents[i], 
                    message_contents[j]
                )
                diversity = 1.0 - similarity  # 유사도의 역수가 다양성
                diversity_scores.append(diversity)
        
        # 평균 다양성 점수
        if diversity_scores:
            ids = np.mean(diversity_scores)
        else:
            ids = 0.0
        
        # 에이전트 타입 다양성 보너스
        agent_types = set(node['agent_type'] for node in graph.nodes)
        type_diversity_bonus = len(agent_types) / 3.0  # 3개 에이전트 타입 (planner, critic, executor)
        
        # 최종 IDS
        ids_final = 0.7 * ids + 0.3 * type_diversity_bonus
        
        return np.clip(ids_final, 0.0, 1.0)
    
    def _calculate_upr(self, graph: AgentInteractionGraph) -> float:
        """
        Unnecessary Path Ratio (UPR) 계산
        
        중복 추론 포착: 불필요하거나 중복된 경로의 비율
        
        논문: "Unnecessary Path Ratio (UPR) captures redundant reasoning"
        """
        if len(graph.nodes) < 2:
            return 0.0
        
        # 모든 경로 찾기
        paths = graph.get_paths()
        
        if not paths:
            return 0.0
        
        # 최적 경로 길이 (Planner → Critic → Executor = 3 노드)
        optimal_path_length = 3
        
        # 불필요한 경로 카운트
        unnecessary_paths = 0
        total_paths = len(paths)
        
        for path in paths:
            path_length = len(path)
            
            # 경로가 최적 길이보다 길면 불필요한 경로로 간주
            if path_length > optimal_path_length:
                unnecessary_paths += 1
            
            # 중복 노드가 있는 경로도 불필요
            if len(path) != len(set(path)):
                unnecessary_paths += 1
        
        # UPR = 불필요한 경로 / 전체 경로
        upr = unnecessary_paths / total_paths if total_paths > 0 else 0.0
        
        # 순차적 협력에서는 UPR이 낮아야 함 (중복이 적어야 함)
        return np.clip(upr, 0.0, 1.0)
    
    def _calculate_jaccard_similarity(self, str1: str, str2: str) -> float:
        """Jaccard 유사도 계산"""
        # 간단한 토큰 기반 Jaccard 유사도
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def is_efficient_collaboration(self, ids: float, upr: float,
                                   ids_threshold=0.6, upr_threshold=0.3) -> bool:
        """
        효율적인 협력 여부 판단
        
        논문: "High IDS and low UPR indicate efficient collaboration"
        
        Args:
            ids: Information Diversity Score
            upr: Unnecessary Path Ratio
            ids_threshold: IDS 임계값 (높아야 함)
            upr_threshold: UPR 임계값 (낮아야 함)
            
        Returns:
            bool: 효율적인 협력 여부
        """
        return ids >= ids_threshold and upr <= upr_threshold
    
    def get_collaboration_feedback(self, ids: float, upr: float) -> str:
        """협력 품질에 대한 피드백 생성"""
        if self.is_efficient_collaboration(ids, upr):
            return f"✅ Efficient collaboration (IDS={ids:.3f}, UPR={upr:.3f})"
        elif ids < 0.6:
            return f"⚠️ Low information diversity (IDS={ids:.3f}). Agents may be redundant."
        elif upr > 0.3:
            return f"⚠️ High unnecessary paths (UPR={upr:.3f}). Reasoning may be redundant."
        else:
            return f"ℹ️ Moderate collaboration quality (IDS={ids:.3f}, UPR={upr:.3f})"

