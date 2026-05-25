import { NextResponse } from "next/server";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import type {
  DrinkFeedback,
  DrinkIngredient,
  GenerationConstraints,
  RateDrinkRequest,
} from "@/types/drink";

type SkillJson = {
  sourceMarkdown: string;
  recipeCount: number;
  recipes: Array<Record<string, unknown>>;
};

type HistoryJson = {
  records: Array<Record<string, unknown>>;
};

const categoryFiles = [
  { key: "season", file: "season.json", markdown: "season.md" },
  { key: "price_band", file: "price_band.json", markdown: "price_band.md" },
  { key: "cost_cap", file: "cost_cap.json", markdown: "cost_cap.md" },
  { key: "make_time", file: "make_time.json", markdown: "make_time.md" },
  {
    key: "target_audience",
    file: "target_audience.json",
    markdown: "target_audience.md",
  },
  { key: "sweetness", file: "sweetness.json", markdown: "sweetness.md" },
  { key: "temperature", file: "temperature.json", markdown: "temperature.md" },
] as const;

function normalizeRating(value: unknown) {
  const rating = Number(value);

  if (!Number.isFinite(rating)) {
    return null;
  }

  return Math.max(0, Math.min(10, rating));
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function estimateSales(averageRating: number, scores: number[]) {
  const spread = Math.max(...scores) - Math.min(...scores);
  const baseCups = Math.round(averageRating * 42);
  const stabilityBonus = Math.max(0, 40 - spread * 6);
  const weeklyCups = Math.max(12, baseCups + stabilityBonus);

  return {
    weeklyCups,
    monthlyCups: weeklyCups * 4,
    conversionRate: Number(Math.min(0.42, 0.08 + averageRating * 0.028).toFixed(2)),
    repeatRate: Number(Math.min(0.58, 0.06 + averageRating * 0.04).toFixed(2)),
  };
}

function normalizeFeedback(value: unknown): DrinkFeedback | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const source = value as Partial<DrinkFeedback>;
  const score = normalizeRating(source.score);

  if (score === null) {
    return null;
  }

  return {
    score,
    comment: String(source.comment ?? "").trim(),
    createdAt: source.createdAt || new Date().toISOString(),
  };
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

function normalizeSeason(season = "") {
  if (/春/.test(season)) {
    return "春夏";
  }
  if (/夏/.test(season)) {
    return "夏季";
  }
  if (/秋|冬/.test(season)) {
    return "秋冬";
  }
  return season || "全年常规";
}

function normalizeSweetness(sweetness = "") {
  if (/无糖/.test(sweetness)) {
    return "无糖";
  }
  if (/低|中低/.test(sweetness)) {
    return "低甜";
  }
  if (/高/.test(sweetness)) {
    return "高甜";
  }
  return sweetness || "标准甜";
}

function sectionForCategory(
  categoryKey: (typeof categoryFiles)[number]["key"],
  constraints?: GenerationConstraints,
) {
  if (categoryKey === "season") {
    return normalizeSeason(constraints?.season);
  }
  if (categoryKey === "price_band") {
    return constraints?.priceBand || "未分类价格带";
  }
  if (categoryKey === "cost_cap") {
    return constraints?.maxIngredientCost || "未分类成本";
  }
  if (categoryKey === "make_time") {
    return constraints?.maxMakeTime || "未分类出杯时间";
  }
  if (categoryKey === "target_audience") {
    return inferAudienceHeading(constraints?.targetAudience);
  }
  if (categoryKey === "sweetness") {
    return normalizeSweetness(constraints?.sweetness);
  }
  return constraints?.temperature || "冰饮";
}

function ingredientsToMarkdown(ingredients: DrinkIngredient[]) {
  return ingredients
    .map((ingredient) => `${ingredient.name} ${ingredient.amount}`)
    .join("；");
}

function buildMarkdownLine(
  record: Record<string, unknown>,
  averageRating: number,
) {
  const problem = record.problem ? `。问题：${record.problem}` : "";
  return `- **${record.name}**（\`${record.id}\`，${record.style}，${record.volume}，${record.sweetness}，用户评分均分 ${averageRating.toFixed(
    1,
  )}）：${ingredientsToMarkdown(record.ingredients as DrinkIngredient[])}${problem}`;
}

function appendMarkdownSection(
  content: string,
  section: string,
  line: string,
) {
  const sectionHeading = `## ${section}`;
  const lines = content.endsWith("\n") ? content.slice(0, -1).split("\n") : content.split("\n");
  const start = lines.findIndex((currentLine) => currentLine.trim() === sectionHeading);

  if (start < 0) {
    return `${content.trimEnd()}\n\n${sectionHeading}\n\n${line}\n`;
  }

  let end = lines.length;
  for (let index = start + 1; index < lines.length; index += 1) {
    if (lines[index].startsWith("## ")) {
      end = index;
      break;
    }
  }

  const nextLines = [...lines];
  nextLines.splice(end, 0, line);
  return `${nextLines.join("\n")}\n`;
}

async function appendToSkillFile(
  libraryDir: string,
  category: (typeof categoryFiles)[number],
  record: Record<string, unknown>,
  section: string,
  averageRating: number,
) {
  const dir = path.join(process.cwd(), "skills", libraryDir);
  await mkdir(dir, { recursive: true });

  const jsonPath = path.join(dir, category.file);
  const markdownPath = path.join(dir, category.markdown);
  const jsonData = JSON.parse(await readFile(jsonPath, "utf-8")) as SkillJson;
  jsonData.recipes.push({ ...record, section });
  jsonData.recipeCount = jsonData.recipes.length;
  await writeFile(jsonPath, `${JSON.stringify(jsonData, null, 2)}\n`, "utf-8");

  const markdown = await readFile(markdownPath, "utf-8");
  const nextMarkdown = appendMarkdownSection(
    markdown,
    section,
    buildMarkdownLine({ ...record, section }, averageRating),
  );
  await writeFile(markdownPath, nextMarkdown, "utf-8");

  return path.join("skills", libraryDir, category.file);
}

async function appendHistoryRecord(record: Record<string, unknown>) {
  const historyDir = path.join(process.cwd(), "data");
  const historyPath = path.join(historyDir, "recipe-history.json");
  await mkdir(historyDir, { recursive: true });

  let history: HistoryJson = { records: [] };
  try {
    history = JSON.parse(await readFile(historyPath, "utf-8")) as HistoryJson;
  } catch {
    history = { records: [] };
  }

  history.records.unshift(record);
  await writeFile(historyPath, `${JSON.stringify(history, null, 2)}\n`, "utf-8");
  return path.join("data", "recipe-history.json");
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as RateDrinkRequest | null;
  const feedbacks = Array.isArray(body?.feedbacks)
    ? body.feedbacks.map(normalizeFeedback)
    : [];

  if (!body?.recipe?.name || !body.recipe.ingredients?.length) {
    return NextResponse.json(
      { message: "缺少待评分的配方信息。" },
      { status: 400 },
    );
  }

  if (!feedbacks.length || feedbacks.some((feedback) => feedback === null)) {
    return NextResponse.json(
      { message: "请至少提交 1 条包含 0-10 分分数的有效反馈。" },
      { status: 400 },
    );
  }

  const normalizedFeedbacks = feedbacks as DrinkFeedback[];
  const scores = normalizedFeedbacks.map((feedback) => feedback.score);
  const averageRating = average(scores);
  const isGood = averageRating > 7;
  const libraryDir = isGood ? "recipe_skill_library" : "bad_recipe_skill_library";
  const idPrefix = isGood ? "user_good" : "user_bad";
  const id = `${idPrefix}_${Date.now()}`;
  const record: Record<string, unknown> = {
    section: "",
    name: body.recipe.name,
    id,
    style: isGood ? "用户高分配方" : "用户低分配方",
    volume: "500ml",
    sweetness: body.constraints?.sweetness || "未标注",
    estimatedCostYuan: null,
    ingredients: body.recipe.ingredients,
    description: body.recipe.description,
    steps: body.recipe.steps,
    feedbacks: normalizedFeedbacks,
    userRatings: scores,
    averageRating: Number(averageRating.toFixed(2)),
  };

  if (!isGood) {
    record.problem = `用户平均评分 ${averageRating.toFixed(1)} 分，低于高分入库阈值 7 分`;
  }

  try {
    const savedTo = await Promise.all(
      categoryFiles.map((category) =>
        appendToSkillFile(
          libraryDir,
          category,
          record,
          sectionForCategory(category.key, body.constraints),
          averageRating,
        ),
      ),
    );
    const historyPath = await appendHistoryRecord({
      id,
      createdAt: new Date().toISOString(),
      recipe: body.recipe,
      constraints: body.constraints ?? null,
      feedbacks: normalizedFeedbacks,
      userRatings: scores,
      averageRating: Number(averageRating.toFixed(2)),
      library: libraryDir,
      sales: estimateSales(averageRating, scores),
      skillFiles: savedTo,
    });

    return NextResponse.json({
      id,
      averageRating: Number(averageRating.toFixed(2)),
      library: libraryDir,
      savedTo: [...savedTo, historyPath],
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "反馈保存失败，请稍后重试。";
    return NextResponse.json({ message }, { status: 500 });
  }
}
