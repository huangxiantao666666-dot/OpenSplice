# OpenSplice Workflow

## 项目玩法流程描述

OpenSplice 是一个基于 AI 的图像物体替换工具，提供两种互补的操作模式，用户无需任何专业图像处理技能即可完成高质量的图像合成。

### 模式一：自由交互放置（Interactive Placement）

适用于创意场景——用户想把任意前景物体（例如一只猫、一个产品）放到背景图的任意位置。

1. **加载背景**：用户上传一张背景图片。
2. **获取前景**：用户上传自己的前景图片，或输入文字描述调用 AI（Qwen-Image-Plus）生成前景图。
3. **自动抠图**：系统自动用 SAM 3 模型对前景图进行中心点分割，选取面积最大的掩码作为前景物体的轮廓，去除前景自带背景。
4. **交互定位**：用户在预览画布上点击以放置前景物体，并通过滑块自由调节旋转角度（-180°~180°）和缩放比例（0.1×~3×）。预览区实时显示前景叠加到背景上的效果。
5. **选择融合模式**：用户从四种融合方式中选择：
   - **羽化 Alpha 融合**：前景像素直接替换背景，仅对轮廓边缘做高斯羽化，保留前景原始色彩。
   - **泊松融合（Mixed Clone）**：基于梯度域的图像融合，平滑边缘同时保留前景与背景的颜色平衡。
   - **泊松融合（Normal Clone）**：保留前景纹理细节，匹配背景光照颜色，适用于换脸等场景。
   - **硬边 Alpha 融合**：直接像素替换，不做任何边缘处理，用于对比参考。
6. **可选的 AI 增强**：
   - **AI 协调**：将粗糙融合结果发送给 Qwen-Image-Edit 模型，自动修复光影不一致、接缝、色彩偏差和透视问题。
   - **Reinhard 色彩迁移**：在 Lab 色彩空间将前景的均值和标准差线性映射到背景的统计分布，使前景色调与背景协调。
   - **simOPA 评分**：使用实验室的物体放置评估模型对合成结果进行 0~1 分的自然度评分。

### 模式二：SAM 3 语义替换（SAM3 Stitch）

适用于目标替换场景——用户想替换背景图中的某个特定物体（例如把人脸换成猪头、把产品换成另一个产品）。

1. **上传图像**：用户上传待修改的背景图。
2. **语义分割**：用户通过三种方式告诉 SAM 3 要分割哪个物体：
   - **文本提示**：输入中文或英文描述（如"穿红色衣服的人"、"the dog"）。
   - **框选提示**：在图像上点击两次绘制包围框。
   - **点选提示**：点击物体的一个或多个位置标记前景。
3. **选择掩码**：如果分割出多个候选物体，用户可以选择要替换的那一个。
4. **获取替换图**：用户上传替换图片或输入文字让 AI 生成。
5. **物体提取**：系统自动用边缘检测（Canny + 形态学闭合 + 中心轮廓筛选）从替换图中裁剪出主体物体，去除其背景。
6. **融合替换**：将提取出的物体缩放至目标掩码的包围框大小，用泊松融合贴入原图。可选 AI 协调进一步优化效果。

### 核心 AI 模型

| 模型 | 用途 | 运行位置 |
|------|------|----------|
| SAM 3 (848M) | 开集语义分割（文本/框/点） | 本地 CPU/GPU |
| Qwen-Image-Plus | AI 前景图像生成 | 阿里云 DashScope API |
| Qwen-Image-Edit-Max | AI 融合结果协调修复 | 阿里云 DashScope API |
| simOPA (11M) | 合成图像自然度评分 | 本地 CPU |
| Reinhard Color Transfer | 前景-背景色彩迁移 | 本地（纯算法） |

---

## 提示词（用于生成工作流示意图）

```
A clean, professional workflow diagram in a horizontal pipeline style with soft blue and white colors. The diagram shows an AI-powered image object replacement system with two parallel workflows.

TOP ROW (labeled "Workflow 1: Interactive Placement"):
Step icons connected by arrows:
[Upload Background] → [Upload/AI-Generate Foreground] → [SAM3 Auto-Segment (remove background)] → [Click to Position + Adjust Rotation/Scale] → [Choose Blend Mode (Feathered Alpha / Poisson)] → [Optional: AI Harmonize, Color Transfer, simOPA Score] → [Final Composite Image]

The illustration should show: a landscape photo as background, a cat image as foreground, the cat silhouette extracted by SAM3, the cat being positioned on the landscape with rotation/scale controls, and the final blended result.

BOTTOM ROW (labeled "Workflow 2: SAM3 Semantic Replacement"):
Step icons connected by arrows:
[Upload Image] → [SAM3 Segmentation (Text/Box/Point prompts)] → [Select Target Mask] → [Upload/AI-Generate Replacement] → [Crop to Object + Resize] → [Poisson Blend into Mask Region] → [Optional: AI Harmonize] → [Final Composite Image]

The illustration should show: a group photo, a person's face highlighted with a mask, a new face image being cropped and resized, and the final face-swapped result.

Style: Modern flat design, minimal text, clean icons, blue gradient color scheme, suitable for a technical course report or academic poster. White background. No realistic photos — use stylized iconography and abstract representations.

Size: 16:9 widescreen, suitable for PowerPoint slide.
```

---

## 模式对比

| 维度 | 模式一：自由交互放置 | 模式二：SAM3 语义替换 |
|------|---------------------|----------------------|
| 定位方式 | 用户自由点击、旋转、缩放 | SAM3 自动定位目标物体掩码 |
| 掩码来源 | 前景图自动分割（中心点提示） | 背景图交互分割（文本/框/点） |
| 适用场景 | 创意合成、产品展示 | 物体替换、人脸交换 |
| 融合区域 | 用户指定位置的任意矩形区域 | SAM3 掩码的精确包围框 |
| 控制粒度 | 连续（位置、角度、比例） | 离散（掩码选择） |
