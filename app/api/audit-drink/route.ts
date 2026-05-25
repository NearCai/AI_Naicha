import { NextResponse } from "next/server";
import type {
  AuditDrinkRequest,
  AuditResult,
  DrinkRecipe,
  GenerationConstraints,
  RecipeFilterReport,
  StoreIngredient,
} from "@/types/drink";
import { createAIClient, getAIModel } from "@/lib/ai-client";
import { parseAIJsonObject } from "@/lib/parse-ai-json";
import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
} from "openai/resources/chat/completions";

const deepSeekReasoningOptions = {
  thinking: { type: "enabled" },
  reasoning_effort: "high",
  stream: false,
} as const;

const marketSignals = [
  "健康化从简单低糖、零糖转向低负担、成分透明和功能化表达。",
  "新茶饮竞争重点从低价内卷转向供应链效率、单店盈利和产品价值创新。",
  "低糖冰茶、冷泡茶、鲜果茶仍适合夏季与年轻客群，强调清爽、真实茶感和低负担。",
  "消费者更关注成分表，低糖、植物基、天然果香和轻乳口感更容易获得正向评价。",
];

type AIClient = NonNullable<ReturnType<typeof createAIClient>>;

async function createDeepSeekCompletion(
  client: AIClient,
  params: ChatCompletionCreateParamsNonStreaming,
) {
  const completion = await client.chat.completions.create({
    ...params,
    ...deepSeekReasoningOptions,
  } as unknown as ChatCompletionCreateParamsNonStreaming);

  return completion as ChatCompletion;
}

function parseLimit(value = "") {
  const match = value.match(/(\d+(?:\.\d+)?)/);
  return match ? Number(match[1]) : Number.POSITIVE_INFINITY;
}

function parseAmount(value = "") {
  const match = value.match(/(\d+(?:\.\d+)?)/);
  return match ? Number(match[1]) : 0;
}

function estimateRecipeCost(recipe: DrinkRecipe, ingredients: StoreIngredient[]) {
  return recipe.ingredients.reduce((sum, item) => {
    const source = ingredients.find((ingredient) => ingredient.name === item.name);
    if (!source) {
      return sum;
    }

    return sum + parseAmount(item.amount) * parseAmount(source.costPerUnit);
  }, 0);
}

function estimateMakeTime(recipe: DrinkRecipe) {
  const baseSeconds = 25;
  const ingredientSeconds = recipe.ingredients.length * 5;
  const stepSeconds = recipe.steps.length * 5;
  const toppingSeconds = recipe.ingredients.some((item) =>
    /冻|珍珠|椰果|小料|布丁|芋圆/.test(item.name),
  )
    ? 12
    : 0;

  return baseSeconds + ingredientSeconds + stepSeconds + toppingSeconds;
}

function filterRecipesByCode(
  recipes: DrinkRecipe[],
  ingredients: StoreIngredient[],
  constraints?: GenerationConstraints,
) {
  const maxCost = parseLimit(constraints?.maxIngredientCost);
  const maxTime = parseLimit(constraints?.maxMakeTime);
  const availableNames = new Set(ingredients.map((ingredient) => ingredient.name));
  const kept: DrinkRecipe[] = [];
  const rejected: RecipeFilterReport["rejected"] = [];

  for (const recipe of recipes) {
    const reasons: string[] = [];
    const estimatedCost = estimateRecipeCost(recipe, ingredients);
    const estimatedTime = estimateMakeTime(recipe);
    const unavailable = recipe.ingredients
      .map((item) => item.name)
      .filter((name) => !availableNames.has(name));

    if (unavailable.length) {
      reasons.push(`包含门店未选原料：${unavailable.join("、")}`);
    }
    if (Number.isFinite(maxCost) && estimatedCost > maxCost) {
      reasons.push(`估算成本 ¥${estimatedCost.toFixed(2)} 超过 ${constraints?.maxIngredientCost}`);
    }
    if (Number.isFinite(maxTime) && estimatedTime > maxTime) {
      reasons.push(`估算出杯时间 ${estimatedTime} 秒超过 ${constraints?.maxMakeTime}`);
    }
    if (recipe.ingredients.length < 4 || recipe.ingredients.length > 8) {
      reasons.push("配料数量不在 4-8 项范围内");
    }
    if (recipe.steps.length < 4 || recipe.steps.length > 6) {
      reasons.push("制作步骤不在 4-6 步范围内");
    }

    if (reasons.length) {
      rejected.push({ name: recipe.name, reasons });
    } else {
      kept.push(recipe);
    }
  }

  const fallbackKept = kept.length ? kept : recipes.slice(0, 3);

  return {
    recipes: fallbackKept,
    report: {
      keptCount: fallbackKept.length,
      rejectedCount: rejected.length,
      rejected,
    },
  };
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

function buildMockAudit(recipes: DrinkRecipe[]): AuditResult {
  const selected = recipes[0];

  return {
    auditorName: "审核员",
    selectedRecipeName: selected?.name ?? "",
    summary:
      "审核员综合市场健康化趋势、低糖清爽偏好、门店可操作性和代码过滤结果，选择该方案作为优先打样款。",
    reasons: [
      "符合低糖、清爽、轻负担的当前茶饮趋势。",
      "使用门店已选原料，制作路径短，适合高峰期出杯。",
      "茶香与果香卖点清晰，适合 18-30 岁尝鲜客群。",
    ],
    marketSignals,
  };
}

function buildAuditorPrompt(
  recipes: DrinkRecipe[],
  filterReport: RecipeFilterReport,
  constraints?: GenerationConstraints,
) {
  return `你叫“审核员”，负责在研发工程师输出并经过代码过滤后的候选配方中，选出最值得优先打样的 1 个配方。

你在审核时需要先“搜索当前市场动态”，以下是本次搜索得到的市场信号摘要：
${marketSignals.map((signal) => `- ${signal}`).join("\n")}

奶茶生成约束：
${formatConstraints(constraints)}

代码过滤报告：
保留 ${filterReport.keptCount} 个，剔除 ${filterReport.rejectedCount} 个。
${filterReport.rejected
  .slice(0, 8)
  .map((item) => `- ${item.name}：${item.reasons.join("；")}`)
  .join("\n") || "无剔除项。"}

候选配方：
${JSON.stringify(recipes, null, 2)}

请严格按照以下 JSON 格式返回，不要输出任何其他内容：
{
  "auditorName": "审核员",
  "selectedRecipeName": "被选中的配方名称，必须来自候选配方",
  "summary": "一句话总结为什么选择它",
  "reasons": ["选择理由1", "选择理由2", "选择理由3"],
  "marketSignals": ["你参考的市场信号1", "你参考的市场信号2"]
}

要求：
- 只允许返回一个 JSON object，禁止 Markdown 代码块、解释文字、前后缀、注释
- JSON 必须能被 JSON.parse 直接解析，所有 key 和字符串都必须使用英文双引号`;
}

async function auditRecipes(
  client: ReturnType<typeof createAIClient>,
  recipes: DrinkRecipe[],
  filterReport: RecipeFilterReport,
  constraints?: GenerationConstraints,
) {
  if (!client) {
    return buildMockAudit(recipes);
  }

  try {
    const completion = await createDeepSeekCompletion(client, {
      model: getAIModel(),
      messages: [
        {
          role: "system",
          content: buildAuditorPrompt(recipes, filterReport, constraints),
        },
      ],
      temperature: 0.45,
      max_tokens: 900,
    });

    const content = completion.choices[0]?.message?.content ?? "";
    const audit = parseAIJsonObject<AuditResult>(content);
    const selectedNames = new Set(recipes.map((recipe) => recipe.name));

    if (
      audit.auditorName !== "审核员" ||
      !selectedNames.has(audit.selectedRecipeName) ||
      !Array.isArray(audit.reasons)
    ) {
      return buildMockAudit(recipes);
    }

    return {
      ...audit,
      marketSignals: audit.marketSignals?.length
        ? audit.marketSignals
        : marketSignals,
    };
  } catch {
    return buildMockAudit(recipes);
  }
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as AuditDrinkRequest | null;

  if (!body?.recipes?.length) {
    return NextResponse.json(
      { message: "缺少待审核的候选配方。" },
      { status: 400 },
    );
  }

  const availableIngredients = Array.isArray(body.availableIngredients)
    ? body.availableIngredients
    : [];
  const filterResult = filterRecipesByCode(
    body.recipes,
    availableIngredients,
    body.constraints,
  );
  const audit = await auditRecipes(
    createAIClient(),
    filterResult.recipes,
    filterResult.report,
    body.constraints,
  );

  return NextResponse.json({
    recipes: filterResult.recipes,
    filterReport: filterResult.report,
    audit,
  });
}
