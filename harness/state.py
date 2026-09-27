"""Agent 单次运行时的状态管理。"""

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """
    保存一次用户问题处理过程中的运行状态。
    """

    # 用户最初的问题
    original_query: str

    # 最多允许调用几次搜索
    max_search_calls: int

    # 已经搜索了几次
    search_count: int = 0

    # 最近一次真正执行的 Query
    last_query: str | None = None

    # 已经搜索过的 Query，用于去重
    used_queries: set[str] = field(default_factory=set)

    # 保存已经拿到的 Evidence
    evidence: list[str] = field(default_factory=list)

    # 保存 Agent 的运行轨迹
    trace: list[dict] = field(default_factory=list)

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        把 Query 统一格式，方便判断是否重复。
        """
        return " ".join(query.lower().split())

    def reserve_search(
        self,
        query: str,
        step: int,
    ) -> tuple[bool, str]:
        """
        判断当前搜索是否允许执行。

        返回：
            (True, "ok")
            或
            (False, 原因)
        """

        normalized_query = self.normalize_query(query)

        if not normalized_query:
            return False, "empty_query"

        # 避免重复 Query
        if normalized_query in self.used_queries:
            return False, "duplicate_query"

        # 搜索次数达到上限
        if self.search_count >= self.max_search_calls:
            return False, "budget_exhausted"

        # 搜索被允许后，占用一次预算
        self.search_count += 1
        self.last_query = query.strip()
        self.used_queries.add(normalized_query)

        self.add_trace(
            "search_accepted",
            step=step,
            query=self.last_query,
        )

        return True, "ok"

    def record_evidence(self, evidence: str) -> None:
        """
        保存一次检索返回的 Evidence。
        """
        self.evidence.append(evidence)

    def add_trace(self, event: str, **details) -> None:
        """
        记录 Agent 运行事件。
        """
        self.trace.append(
            {
                "event": event,
                **details,
            }
        )
