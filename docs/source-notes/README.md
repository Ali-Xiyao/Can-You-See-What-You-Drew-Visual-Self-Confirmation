# Source notes

本目录保存实质性改变了实验设计的用户输入原件。它们是溯源记录；规范性要求在 proposal 正文中，
当前状态与锁定选择在 `STATUS.md` 中，实测证据在 `docs/EVIDENCE_LOG.md` 中。

- `model-backbone-revision-20260828.txt`: 用户提供的 backbone/模型选型说明原文，
  SHA-256 `32fddabacaf8bc84c5ed9bba41e0b20af9f6f9e5ef263996865ac8e9ce03c41a`。
  确立了核心约束：同一个可训练模型既能生成图像，又能重新读取 RGB 并回答视觉问题。
