"""User-triggered local memory management, outside the UI thread."""
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class MemoryDialog(QDialog):
    def __init__(self, parent, memory, owner):
        super().__init__(parent)
        self.memory, self.owner = memory, owner
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.busy = False
        self.setWindowTitle("本地长期记忆")
        self.resize(650, 450)
        layout = QVBoxLayout(self)
        self.status = QLabel("记忆只保存在本机，按用户隔离。删除会话不会自动删除长期记忆；清空记忆不会修改个人偏好表。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.items = QListWidget()
        layout.addWidget(self.items)
        controls = QHBoxLayout()
        self.buttons = []
        for text, handler in [("刷新", self.refresh), ("删除选中记忆", self.delete_selected), ("清空当前用户记忆", self.clear)]:
            button = QPushButton(text)
            button.clicked.connect(handler)
            controls.addWidget(button)
            self.buttons.append(button)
        layout.addLayout(controls)
        self.refresh()

    def run(self, operation, callback):
        if self.busy:
            return
        from stylemate.desktop.application import Task
        self.busy = True
        for button in self.buttons:
            button.setEnabled(False)
        self.status.setText("正在读取本地记忆…")
        task = Task(operation)
        self.task = task
        def completed(value):
            self.busy = False
            for button in self.buttons:
                button.setEnabled(True)
            callback(value)
        task.signals.done.connect(completed)
        task.signals.failed.connect(lambda _: completed(None))
        self.pool.start(task)

    def refresh(self):
        self.run(lambda: self.memory.list(self.owner), self.show_items)

    def show_items(self, values):
        if values is None:
            self.status.setText("本地记忆操作未完成，请检查数据目录权限后重试。")
            return
        self.items.clear()
        for value in values:
            item = QListWidgetItem(str(value.get("memory", "")))
            item.setData(Qt.ItemDataRole.UserRole, value.get("id"))
            self.items.addItem(item)
        self.status.setText(f"共 {len(values)} 条本地记忆。已删除的相同内容不会在后续对话中自动恢复。")

    def delete_selected(self):
        item = self.items.currentItem()
        if item is None:
            return
        if QMessageBox.question(self, "删除记忆", "确定删除选中的长期记忆？") != QMessageBox.StandardButton.Yes:
            return
        memory_id = item.data(Qt.ItemDataRole.UserRole)
        def operation():
            self.memory.delete(self.owner, memory_id)
            return self.memory.list(self.owner)
        self.run(operation, self.show_items)

    def clear(self):
        if QMessageBox.question(self, "清空记忆", "确定清空当前用户的长期记忆？衣橱、个人偏好表与本地会话会保留。") != QMessageBox.StandardButton.Yes:
            return
        def operation():
            self.memory.clear(self.owner)
            return self.memory.list(self.owner)
        self.run(operation, self.show_items)

    def reject(self):
        if not self.busy:
            super().reject()
