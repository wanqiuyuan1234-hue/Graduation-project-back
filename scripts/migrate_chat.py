"""一次性迁移：创建对话会话与消息两张表。

运行：在项目根目录执行 ``.venv\\Scripts\\python.exe scripts/migrate_chat.py``

- 只创建 `chat_session` / `chat_message` 及其索引，可重复执行（checkfirst）；
- 不修改既有 repo / code_file / chunk 表；
- 删除会话时消息由外键 ON DELETE CASCADE 级联清理。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect

from app.database import Base, engine
from app.models import ChatMessage, ChatSession


def main() -> None:
    Base.metadata.create_all(
        bind=engine,
        tables=[ChatSession.__table__, ChatMessage.__table__],
        checkfirst=True,
    )

    inspector = inspect(engine)
    for table in ("chat_session", "chat_message"):
        if inspector.has_table(table):
            columns = ", ".join(column["name"] for column in inspector.get_columns(table))
            print(f"已就绪：{table}（{columns}）")
        else:
            print(f"创建失败：{table}")
            sys.exit(1)

    print("对话表迁移完成")


if __name__ == "__main__":
    main()
