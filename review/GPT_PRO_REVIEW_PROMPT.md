# Prompt for GPT Pro

请打开并审阅这个 GitHub 仓库。先阅读 `paper/main.pdf` 和 `paper/supplementary.pdf`，再使用 `review/CLAIM_EVIDENCE_MAP.md`、`data/recording_catalog_and_splits.csv`、`data/RAW_DATA_PROVENANCE.md`、`review/SESSION_CONFOUNDING_AUDIT.md`、`evidence/` 以及必要的源代码核查论文主张。

请以机械故障诊断、旋转机械状态监测和工程类 SCI 期刊审稿人的标准进行独立审阅，而不是只做英语润色。重点检查：

1. 研究问题、创新定位和目标期刊层级是否匹配；
2. recording-grouped 划分是否足以控制同一采集记录的窗口泄漏；
3. 是否把 acquisition recording 错当成独立物理轴承实例；
4. 84 个原始文件的 SHA-256/形状审计和重复时间戳深度比较，是否充分排除了“同一文件重复上传”这一具体疑点，同时没有被夸大成 specimen/session independence；
5. acquisition date 与类别的描述性关联是否被正确解释为 session-confounding risk，而不是新的诊断准确率；
6. cross-condition、cross-load 和 cross-RPM 的方向性实验是否定义清楚；
7. 训练随机种子、重复数据划分和 recording-cluster bootstrap 三类不确定性是否区分正确；
8. 窗口级结果是否由四套主要协议的经典模型 recording-level aggregation 与 bootstrap 充分补充；
9. 摘要、表格、图、讨论和结论中的数字是否能追溯到证据 CSV；
10. exact feature membership、order-related features 和 fusion-input semantics 的解释是否合理；
11. `auxiliary_26` 与 `auxiliary_context_28` 是否只保留在补充材料并始终被视为同一 3000-rpm holdout 上的 post-hoc exploratory evidence，而非确认性优越结果；
12. 机械机理解释是否超过现有数据能够支持的范围；
13. 代码、数据说明和复现材料是否足以支撑公开审阅。

请按以下格式输出：

- 总体决定：Accept / Minor Revision / Major Revision / Reject；
- 主要问题：按严重程度排序，每条指出论文页码、章节、表图或证据文件；
- 次要问题：语言、格式、术语、图表和参考文献问题；
- 主张—证据一致性检查；
- 统计与实验设计检查；
- 机械工程解释检查；
- 投稿风险与适合的期刊类型；
- 投稿前必须修改的最小清单。

请特别遵守以下证据边界：15,036 个重叠窗口不是独立实验重复；独立 SHA-256 只排除完全相同的文件，不能证明不同物理轴承或独立采集会话；采集日期关联是描述性混杂审计，不是预测实验；论文不提供跨机器或非 MCC5 泛化证据；辅助 26/28 输入配置是后验探索结果。不要因为仓库提供了大量代码就默认算法创新已经成立。
