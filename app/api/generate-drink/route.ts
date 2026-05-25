import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";
import type {
  GenerateDrinkRequest,
  DrinkRecipe,
  DrinkDevelopmentResult,
  GenerationConstraints,
  StoreProfile,
  StoreIngredient,
} from "@/types/drink";
import {
  createAIClient,
  getAICompletionOptions,
  getAIModel,
} from "@/lib/ai-client";
import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
} from "openai/resources/chat/completions";
import { parseAIJsonObject } from "@/lib/parse-ai-json";

type AIClient = NonNullable<ReturnType<typeof createAIClient>>;

async function createDeepSeekCompletion(
  client: AIClient,
  params: ChatCompletionCreateParamsNonStreaming,
) {
  const completion = await client.chat.completions.create({
    ...params,
    ...getAICompletionOptions(),
  } as unknown as ChatCompletionCreateParamsNonStreaming);

  return completion as ChatCompletion;
}

const skillFiles = [
  { file: "season.md", label: "季节" },
  { file: "price_band.md", label: "价格带" },
  { file: "cost_cap.md", label: "成本上限" },
  { file: "make_time.md", label: "出杯时间" },
  { file: "target_audience.md", label: "目标人群" },
  { file: "sweetness.md", label: "甜度倾向" },
  { file: "temperature.md", label: "温度形态" },
];

const mockRecipeBase: DrinkRecipe = {
  name: "青提茉莉轻乳茶",
  description:
    "以茉莉茶香承接青提果香，轻乳降低厚重感，整体清爽明亮，适合夏季年轻客群。",
  ingredients: [
    { name: "茉莉绿茶汤", amount: "180g" },
    { name: "青提果汁", amount: "60g" },
    { name: "轻乳基底", amount: "80g" },
    { name: "冰块", amount: "120g" },
    { name: "茉莉茶冻", amount: "45g" },
    { name: "蔗糖糖浆", amount: "8g" },
  ],
  steps: [
    "杯中加入青提果汁和茉莉绿茶汤。",
    "加入轻乳基底、蔗糖糖浆和冰块后充分摇匀。",
    "倒入出品杯中。",
    "加入茉莉茶冻增强茶香层次。",
    "封口后轻摇两次，立即出品。",
  ],
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const defaultGenerationCount = 1;
const maxGenerationCount = 6;

function normalizeGenerationCount(value: unknown) {
  const count = Number(value);

  if (!Number.isFinite(count)) {
    return defaultGenerationCount;
  }

  return Math.max(1, Math.min(maxGenerationCount, Math.round(count)));
}

async function loadSkillLibrary(constraints?: GenerationConstraints) {
  const skillDir = path.join(process.cwd(), "skills", "recipe_skill_library");
  const loaded = await Promise.all(
    skillFiles.map(async ({ file, label }) => {
      const content = await readFile(path.join(skillDir, file), "utf-8");
      return {
        label,
        file,
        content: pickRelevantSkillSections(content, constraints, 6),
      };
    }),
  );

  return {
    references: loaded.map((item) => `${item.label}:${item.file}`),
    content: loaded
      .map((item) => `# ${item.label} Skill（${item.file}）\n${item.content}`)
      .join("\n\n---\n\n"),
  };
}

function pickRelevantSkillSections(
  content: string,
  constraints?: GenerationConstraints,
  maxRecipesPerSection = 8,
) {
  const headings = [
    constraints?.season,
    constraints?.priceBand,
    constraints?.maxIngredientCost,
    constraints?.maxMakeTime,
    constraints?.sweetness,
    constraints?.temperature,
    inferAudienceHeading(constraints?.targetAudience),
  ].filter(Boolean) as string[];

  const sections = new Set<string>();
  for (const heading of headings) {
    const section = extractMarkdownSection(content, heading);
    if (section) {
      sections.add(limitMarkdownSection(section, maxRecipesPerSection));
    }
  }

  if (!sections.size) {
    return content.split("\n").slice(0, 18).join("\n");
  }

  return Array.from(sections).join("\n\n");
}

function limitMarkdownSection(section: string, maxRecipes: number) {
  const lines = section.split("\n");
  const heading = lines[0];
  const recipeLines = lines.filter((line) => line.startsWith("- "));

  if (recipeLines.length) {
    return [heading, ...recipeLines.slice(0, maxRecipes)].join("\n");
  }

  return lines.slice(0, Math.min(lines.length, maxRecipes + 1)).join("\n");
}

function extractMarkdownSection(content: string, heading: string) {
  const lines = content.split("\n");
  const start = lines.findIndex((line) => line.trim() === `## ${heading}`);

  if (start < 0) {
    return "";
  }

  let end = lines.length;
  for (let i = start + 1; i < lines.length; i += 1) {
    if (lines[i].startsWith("## ")) {
      end = i;
      break;
    }
  }

  return lines.slice(start, end).join("\n");
}

function inferAudienceHeading(targetAudience = "") {
  if (/健康|轻负担|低糖|低卡|控糖|无负担/.test(targetAudience)) {
    return "健康轻负担人群";
  }
  if (/上班|提神|咖啡|职场/.test(targetAudience)) {
    return "上班族提神人群";
  }
  if (/年轻|尝鲜|学生|18|30|果香/.test(targetAudience)) {
    return "年轻尝鲜人群";
  }
  if (/珍珠|小料|布丁|芋圆|黑糖/.test(targetAudience)) {
    return "经典小料偏好人群";
  }
  return "大众日常人群";
}

function formatIngredientLibrary(ingredients: StoreIngredient[]) {
  if (!ingredients.length) {
    return "未提供门店原料库。";
  }

  return ingredients
    .map((ingredient) => {
      const tags = ingredient.flavorTags.join("、") || "无";
      const allergens = ingredient.allergens.join("、") || "无";
      const equipment = ingredient.equipment.join("、") || "无特殊设备";

      return `- ${ingredient.id}｜${ingredient.name}｜类别：${ingredient.category}｜数量：${ingredient.quantity}｜成本：${ingredient.costPerUnit}｜风味：${tags}｜过敏原：${allergens}｜供应：${ingredient.availability}｜设备：${equipment}`;
    })
    .join("\n");
}

function formatStoreProfile(storeProfile?: StoreProfile) {
  if (!storeProfile) {
    return "未提供门店基础信息。";
  }

  const equipment = storeProfile.equipment.length
    ? storeProfile.equipment.join("、")
    : "未选择";

  return [
    `门店名称：${storeProfile.storeName || "未填写"}`,
    `门店类型：${storeProfile.storeType || "未填写"}`,
    `品牌风格：${storeProfile.brandStyle || "未填写"}`,
    `可用设备：${equipment}`,
  ].join("\n");
}

function formatConstraints(constraints?: GenerationConstraints) {
  if (!constraints) {
    return "未提供奶茶生成约束。";
  }

  return [
    `季节：${constraints.season || "未填写"}`,
    `目标人群：${constraints.targetAudience || "未填写"}`,
    `价格带：${constraints.priceBand || "未填写"}`,
    `单杯原料成本上限：${constraints.maxIngredientCost || "未填写"}`,
    `出杯时间上限：${constraints.maxMakeTime || "未填写"}`,
    `甜度倾向：${constraints.sweetness || "未填写"}`,
    `温度形态：${constraints.temperature || "未填写"}`,
  ].join("\n");
}

function buildMockRecipes(ingredients: StoreIngredient[], generationCount = 1): DrinkRecipe[] {
  if (!ingredients.length) {
    return Array.from({ length: generationCount }, (_, index) => ({
      ...mockRecipeBase,
      name: `${mockRecipeBase.name}${index + 1}`,
    }));
  }

  const selected = ingredients.slice(0, 6);
  const names = [
    "青提茉莉轻乳茶",
    "青提茶冻云乳",
    "茉莉青提椰果茶",
    "青提轻乳冰茶",
    "茉莉葡香小白杯",
    "青提云雾奶茶",
    "茉莉茶冻轻奶",
    "青提鲜乳雪克",
    "清茉青提乳茶",
    "青提茉莉冰乳",
  ];

  return names.slice(0, generationCount).map((name, recipeIndex) => ({
    name,
    description:
      "研发工程师参考配方 skill 库后，基于当前门店原料生成的候选方案，强调清爽果香、茶感和轻负担口感。",
    ingredients: selected.slice(0, 5 + (recipeIndex % 2)).map((ingredient, index) => ({
      name: ingredient.name,
      amount: mockAmountForIngredient(ingredient, index),
    })),
    steps: [
      "按门店 SOP 备好茶底、奶基底和风味原料。",
      "将茶底、风味原料和糖浆加入雪克杯。",
      "加入奶基底和冰块后充分摇匀。",
      "倒入出品杯，加入小料形成口感层次。",
      "封口后轻摇两次，立即出品。",
    ],
  }));
}

function mockAmountForIngredient(ingredient: StoreIngredient, index: number) {
  if (ingredient.category === "茶底") {
    return `${100 + (index % 2) * 10}g`;
  }
  if (ingredient.category === "奶基底") {
    return "35g";
  }
  if (ingredient.category === "水果") {
    return "22g";
  }
  if (ingredient.category === "小料") {
    return "15g";
  }
  if (ingredient.category === "风味糖浆") {
    return "5g";
  }
  if (ingredient.category === "辅料") {
    return "80g";
  }
  return "适量";
}

function isCompleteRecipe(recipe: DrinkRecipe) {
  return (
    Boolean(recipe?.name) &&
    Boolean(recipe.description) &&
    Array.isArray(recipe.ingredients) &&
    Array.isArray(recipe.steps) &&
    recipe.ingredients.length >= 4 &&
    recipe.steps.length >= 4
  );
}

function normalizeGeneratedResult(
  result: DrinkDevelopmentResult,
  fallbackRecipes: DrinkRecipe[],
  generationCount: number,
): DrinkDevelopmentResult {
  const recipes = Array.isArray(result.recipes)
    ? result.recipes.filter(isCompleteRecipe)
    : [];
  const seenNames = new Set<string>();
  const deduped = recipes.filter((recipe) => {
    if (seenNames.has(recipe.name)) {
      return false;
    }
    seenNames.add(recipe.name);
    return true;
  });
  const merged = [...deduped];

  for (const fallbackRecipe of fallbackRecipes) {
    if (merged.length >= generationCount) {
      break;
    }
    if (!seenNames.has(fallbackRecipe.name)) {
      merged.push(fallbackRecipe);
      seenNames.add(fallbackRecipe.name);
    }
  }

  return {
    engineerName: "研发工程师",
    skillReferences: result.skillReferences?.length
      ? result.skillReferences
      : [],
    recipes: merged.slice(0, generationCount),
  };
}

function buildSystemPrompt(
  ingredients: StoreIngredient[],
  skillContent: string,
  generationCount: number,
  storeProfile?: StoreProfile,
  constraints?: GenerationConstraints,
): string {
  return `你叫“研发工程师”，是一位专业的现代茶饮研发师，负责根据门店能力、生成约束和配方 skill 库设计新品候选。

工作流程必须是：
1. 先阅读“配方 Skill 库参考”中的现有配方；
2. 再结合门店现有原料库、门店设备和用户需求；
3. 最后输出 ${generationCount} 个可执行奶茶候选配方。

你必须基于以下门店现有原料库设计配方，ingredients 中只能使用原料库里出现的原料名称，不要编造新原料。
如果用户需求和原料库冲突，请优先使用风味相近的现有原料替代，并在 description 中自然说明风味方向。

门店基础信息：
${formatStoreProfile(storeProfile)}

奶茶生成约束：
${formatConstraints(constraints)}

门店现有原料库：
${formatIngredientLibrary(ingredients)}

配方 Skill 库参考：
${skillContent}

请严格按照以下 JSON 格式返回，不要输出任何其他内容：
{
  "engineerName": "研发工程师",
  "skillReferences": ["读取过的 skill 名称"],
  "recipes": [
    {
      "name": "饮品名称（10字以内，有吸引力）",
      "description": "饮品描述（50-80字，突出参考 skill、风味、口感、适合场景）",
      "ingredients": [
        { "name": "原料名称", "amount": "用量（如 180g / 60ml / 2勺）" }
      ],
      "steps": [
        "制作步骤1",
        "制作步骤2"
      ]
    }
  ]
}

要求：
- 只允许返回一个 JSON object，禁止 Markdown 代码块、解释文字、前后缀、注释
- JSON 必须能被 JSON.parse 直接解析，所有 key 和字符串都必须使用英文双引号
- recipes 必须正好 ${generationCount} 个，名称不能重复
- 每个 recipes[i].ingredients 至少 4 项，不超过 8 项
- 每个 recipes[i].steps 至少 4 步，不超过 6 步
- 用量要具体、可执行
- 优先使用供应状态为“充足”的原料，谨慎使用“偏低”和“季节限定”的原料
- 避免不必要的复杂设备和过长制作流程
- 尽量匹配门店类型、品牌风格、目标人群、价格带、成本上限和出杯时间上限
- 只能设计门店设备可以完成的 SOP
- 名称要有创意，避免俗套
- 描述要突出风味层次和卖点`;
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as
    | GenerateDrinkRequest
    | null;

  if (!body?.prompt?.trim()) {
    return NextResponse.json(
      { message: "请输入饮品需求描述。" },
      { status: 400 },
    );
  }

  const availableIngredients = Array.isArray(body.availableIngredients)
    ? body.availableIngredients
    : [];
  const storeProfile = body.storeProfile;
  const constraints = body.constraints;
  const generationCount = normalizeGenerationCount(body.generationCount);
  const skillLibrary = await loadSkillLibrary(constraints);

  const client = createAIClient();

  if (!client) {
    // 未配置 AI API，使用 Mock（延迟模拟真实请求）
    if (body.prompt.includes("失败")) {
      return NextResponse.json(
        { message: "Agent 暂时无法完成这条需求，请换一种描述再试。" },
        { status: 500 },
      );
    }

    await sleep(1200);
      const result: DrinkDevelopmentResult = {
        engineerName: "研发工程师",
        skillReferences: skillLibrary.references,
        recipes: buildMockRecipes(availableIngredients, generationCount),
      };
    return NextResponse.json(result);
  }

  try {
    const completion = await createDeepSeekCompletion(client, {
      model: getAIModel(),
      messages: [
        {
          role: "system",
          content: buildSystemPrompt(
            availableIngredients,
            skillLibrary.content,
            generationCount,
            storeProfile,
            constraints,
          ),
        },
        { role: "user", content: body.prompt.trim() },
      ],
      response_format: { type: "json_object" },
      temperature: 0.8,
      max_tokens: Math.min(5200, 900 + generationCount * 800),
    });

    const content = completion.choices[0]?.message?.content ?? "";

    let result: DrinkDevelopmentResult;
    try {
      result = parseAIJsonObject<DrinkDevelopmentResult>(content);
    } catch {
      result = {
        engineerName: "研发工程师",
        skillReferences: skillLibrary.references,
        recipes: buildMockRecipes(availableIngredients, generationCount),
      };
    }

    result = normalizeGeneratedResult(
      result,
      buildMockRecipes(availableIngredients, generationCount),
      generationCount,
    );

    return NextResponse.json({
      ...result,
      skillReferences: result.skillReferences?.length
        ? result.skillReferences
        : skillLibrary.references,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "配方生成失败，请稍后重试。";
    return NextResponse.json({ message }, { status: 500 });
  }
}
