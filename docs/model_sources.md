# 模型来源和复用方式

2026-10-04 检索并查验了公开海事仿真仓库 [OSRF VRX](https://github.com/osrf/vrx)，固定参考提交 `41f2df50fbf75b30264eb4c7a2222d4235f6a70b`。仓库根 LICENSE 为 Apache-2.0，查看了 roboboat01/model.sdf 及其 DAE/PBR 纹理结构。它提供的是小型无人船/WAM-V，**不是护卫舰**；本项目参考其视觉网格与物理模型分离、独立传感器外参和 PBR 材质配置，不把小船模型声称为护卫舰。

开源船模参考：

- [roboboat01 SDF](https://github.com/osrf/vrx/blob/41f2df50fbf75b30264eb4c7a2222d4235f6a70b/vrx_gz/models/roboboat01/model.sdf)
- [WAM-V 资产目录](https://github.com/osrf/vrx/tree/41f2df50fbf75b30264eb4c7a2222d4235f6a70b/vrx_urdf/wamv_description/models)
- [VRX Apache-2.0 LICENSE](https://github.com/osrf/vrx/blob/41f2df50fbf75b30264eb4c7a2222d4235f6a70b/LICENSE)

GitHub 中检索 frigate / warship / destroyer 模型时，部分候选是游戏导出工具或没有许可证的游戏 MOD，未发现可直接确认许可并适合本任务的护卫舰网格，因此没有复制它们的资产。当前默认护卫舰是本项目原创参数化基线：尖艏闭合船体、机库、舰桥窗、桅杆、雷达、烟囱和艉部 H 着陆甲板；它不对应真实型号，也不宣称真实水动力精度。

`visual_modeling/assets/ships/sources.json` 记录可机读来源；`shell/fetch_reference_assets.sh` 可选下载固定提交的 VRX 小船 DAE/SDF 与 LICENSE 供设计参考。默认护卫舰渲染不依赖这些参考文件。要使用其他公开护卫舰资产，应先确认网格/纹理各自许可，在 sources.json 记录链接、作者、许可及变更，再转换为 GLB/OBJ/Blend 并设置 scene.yaml 的 ship.asset_path。导入前需把模型长度方向对齐 +X；导入器统一长度并添加独立 H 甲板。第三方船的机库/甲板位置可能需要在原资产中预先对齐，模型几何遮挡通过标注可见性反映。
