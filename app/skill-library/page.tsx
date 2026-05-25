import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  SkillLibraryView,
  type SkillCategory,
  type SkillLibrary,
} from "./skill-library-view";

export const dynamic = "force-dynamic";

const categoryFiles = [
  { key: "season", label: "季节", file: "season.json" },
  { key: "price_band", label: "价格带", file: "price_band.json" },
  { key: "cost_cap", label: "成本上限", file: "cost_cap.json" },
  { key: "make_time", label: "出杯时间", file: "make_time.json" },
  { key: "target_audience", label: "目标人群", file: "target_audience.json" },
  { key: "sweetness", label: "甜度倾向", file: "sweetness.json" },
  { key: "temperature", label: "温度形态", file: "temperature.json" },
];

async function loadCategory(
  libraryDir: string,
  category: (typeof categoryFiles)[number],
): Promise<SkillCategory> {
  const filePath = path.join(process.cwd(), "skills", libraryDir, category.file);
  const raw = await readFile(filePath, "utf-8");
  const parsed = JSON.parse(raw) as Omit<SkillCategory, "key" | "label">;

  return {
    ...parsed,
    key: category.key,
    label: category.label,
  };
}

async function loadLibrary(
  key: SkillLibrary["key"],
  libraryDir: string,
  title: string,
  subtitle: string,
): Promise<SkillLibrary> {
  const categories = await Promise.all(
    categoryFiles.map((category) => loadCategory(libraryDir, category)),
  );

  return {
    key,
    title,
    subtitle,
    tone: key,
    categories,
  };
}

export default async function SkillLibraryPage() {
  const libraries = await Promise.all([
    loadLibrary(
      "good",
      "recipe_skill_library",
      "好配方库",
      "从参考配方整理出的正向样本，用来学习稳定、可售、风味清晰的奶茶结构。",
    ),
    loadLibrary(
      "bad",
      "bad_recipe_skill_library",
      "差配方库",
      "人为构造的反面样本，用来识别难喝组合、定位矛盾和不可执行的出品风险。",
    ),
  ]);

  return <SkillLibraryView libraries={libraries} />;
}
