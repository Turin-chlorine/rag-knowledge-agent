"""
会话记忆模块（多轮对话）
以 session_id 为单位保存对话历史，用于：
1. 问题改写：把含指代/省略的追问补全为可独立检索的问题；
2. 上下文生成：将最近若干轮历史随请求发给大模型，保持多轮语义连贯。

内存存储（进程级单例），重启后清空——简历项目规模足够，避免引入 Redis 等外部依赖。
"""
import threading
import uuid

from config import MEMORY_MAX_TURNS


class SessionManager:
    """
    会话管理器：维护多个会话的消息历史，线程安全。
    每个会话只保留最近 MEMORY_MAX_TURNS 轮（一问一答为一轮）。
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS):
        """
        初始化会话管理器。

        :param max_turns: 每个会话保留的最大轮数
        """
        self.max_turns = max_turns
        # session_id -> [{"role": "user"/"assistant", "content": str}, ...]
        self._sessions: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def create_session(self) -> str:
        """
        创建一个新会话。

        :return: 新生成的会话 ID
        """
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = []
        return session_id

    def get_history(self, session_id: str) -> list[dict]:
        """
        获取指定会话最近若干轮的历史消息（用于问题改写与生成上下文）。

        :param session_id: 会话 ID
        :return: 历史消息列表（按时间升序）；会话不存在时返回空列表
        """
        with self._lock:
            messages = self._sessions.get(session_id, [])
            # 仅保留最近 max_turns 轮（2*max_turns 条消息）
            return list(messages[-2 * self.max_turns:])

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """
        向指定会话追加一条消息，并在超出保留轮数时裁剪旧消息。

        :param session_id: 会话 ID
        :param role: 消息角色，user 或 assistant
        :param content: 消息文本
        """
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.append({"role": role, "content": content})
            # 裁剪到最近 max_turns 轮，控制内存与后续请求的 token 长度
            keep_count = 2 * self.max_turns
            if len(messages) > keep_count:
                del messages[:-keep_count]

    def clear_session(self, session_id: str) -> bool:
        """
        清空指定会话的历史（开始新对话时调用）。

        :param session_id: 会话 ID
        :return: True 表示会话存在且已清空，False 表示会话不存在
        """
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id] = []
                return True
            return False


# 全局会话管理器单例
session_manager = SessionManager()
