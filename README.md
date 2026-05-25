# AI 奶茶配方生成

一个面向新茶饮门店的 AI 新品研发原型。系统基于门店原料库、门店设备、生成约束和配方 skill 库，让“研发工程师”生成候选配方，再由代码过滤器和“审核员”完成可行性审核，并为最终配方生成产品图。

## 功能

- 门店配置：维护门店基础信息、设备能力和可用原料。
- 研发模块：研发工程师读取 `skills/recipe_skill_library`，基于当前约束生成 1 个候选配方。
- 配料表查看：研发模块结束后可查看候选配方配料表。
- 审核模块：自动进行代码过滤，检查原料合法性、成本、出杯时间和 SOP 完整度。
- 审核员选择：基于市场信号、商业可行性和产品表达给出审核结论。
- 产品图生成：审核完成后调用火山方舟图片模型为最终配方生成奶茶图片。

## 技术栈

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- OpenAI SDK 兼容 DeepSeek 文本模型
- 火山方舟图片生成接口

## 快速开始

```bash
npm install
cp .env.example .env.local
npm run dev
```

打开本地服务地址，例如 `http://localhost:3000`。如果端口被占用，Next.js 会自动切到其他端口。

## 环境变量

在 `.env.local` 中配置真实密钥。不要提交 `.env.local`。

```env
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING_ENABLED=false
DEEPSEEK_REASONING_EFFORT=medium

IMAGE_API_KEY=your-volcengine-ark-api-key
IMAGE_API_URL=https://ark.cn-beijing.volces.com/api/v3/images/generations
IMAGE_MODEL=doubao-seedream-5-0-260128
IMAGE_SIZE=2K
IMAGE_WATERMARK=true
```

未配置文本模型时，配方生成会走 mock 数据；未配置图片模型时，产品图会走 mock 图片。

研发链路默认使用 `deepseek-v4-flash`，并关闭 thinking/reasoning 参数以缩短生成耗时。如需更强推理，可将 `DEEPSEEK_THINKING_ENABLED` 设置为 `true`。

## 常用命令

```bash
npm run dev
npm run lint
npm run build
npm run start
```

## 目录结构

```text
app/
  api/
    generate-drink/        研发工程师生成配方
    audit-drink/           代码过滤与审核员审核
    generate-drink-image/  产品图生成
  store-config/            门店配置页
components/                UI 组件
lib/                       API 客户端、配置和解析工具
skills/recipe_skill_library/
  season.md
  price_band.md
  cost_cap.md
  make_time.md
  target_audience.md
  sweetness.md
  temperature.md
types/                     业务类型定义
```

## 工作流

1. 在门店配置页选择门店已有原料和设备。
2. 在首页填写季节、价格带、成本上限、出杯时间、目标人群、甜度倾向和温度形态。
3. 研发工程师读取 skill 库并生成 1 个配方。
4. 查看配料表。
5. 系统自动进入审核模块，进行代码过滤和审核员评价。
6. 审核完成后生成最终奶茶产品图。

## 注意事项

- `.env.local` 已被 `.gitignore` 忽略，真实 API key 不应进入 git。
- `skills/recipe_skill_library` 是配方参考知识库，后续可以继续扩展更多品类、季节和客群样本。
- 当前市场信号是内置摘要，后续可以替换为实时搜索或外部数据源。
