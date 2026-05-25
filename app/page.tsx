"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  BookOpen,
  Brain,
  Clipboard,
  FlaskConical,
  Loader2,
  Search,
  Send,
  Settings,
  Sparkles,
  Store,
  TableProperties,
  WandSparkles,
} from "lucide-react";
import { DrinkImageCard, RecipeCard } from "@/components/recipe-card";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  auditDrinkRecipes,
  generateDrinkImage,
  generateDrinkRecipe,
} from "@/lib/api";
import {
  getDefaultStoreConfig,
  ingredientLibrary,
  normalizeStoreConfig,
  storeConfigStorageKey,
} from "@/lib/store-config";
import type { StoreConfig } from "@/lib/store-config";
import type {
  DrinkRecipe,
  DrinkAuditStageResult,
  DrinkDevelopmentResult,
  DrinkGenerationResult,
  GenerationConstraints,
  GenerationStatus,
} from "@/types/drink";

const defaultPrompt =
  "我想要一杯适合夏天、清爽、带有青提和茉莉茶香的奶茶";

const recipeProgressMessages = [
  "正在阅读 skill",
  "搜索信息中",
  "绞尽脑汁中",
  "校验门店约束",
  "调配候选配方",
  "输出 1 个方案",
];

const auditProgressMessages = [
  "代码过滤成本与出杯时间",
  "审核员正在搜索市场动态",
  "审核员综合评价配方",
  "审核员选择最佳方案",
];

const developmentStages = [
  {
    role: "研发工程师",
    icon: BookOpen,
    title: "正在阅读 skill",
    detail: "检索季节、价格带、成本、出杯时间等参考配方。",
  },
  {
    role: "研发工程师",
    icon: Search,
    title: "搜索信息中",
    detail: "在配方库里寻找和当前需求最接近的风味结构。",
  },
  {
    role: "研发工程师",
    icon: Brain,
    title: "绞尽脑汁中",
    detail: "平衡茶感、果香、奶感、甜度和门店可操作性。",
  },
  {
    role: "研发工程师",
    icon: FlaskConical,
    title: "校验门店约束",
    detail: "核对原料、设备、成本上限、出杯时间和温度形态。",
  },
  {
    role: "研发工程师",
    icon: Sparkles,
    title: "调配候选配方",
    detail: "组合 1 个差异化方案，避免重复和不可执行配方。",
  },
  {
    role: "研发工程师",
    icon: WandSparkles,
    title: "输出 1 个方案",
    detail: "整理配料克重、制作流程和每杯的研发描述。",
  },
];

const auditStages = [
  {
    role: "代码过滤器",
    icon: FlaskConical,
    title: "代码过滤成本与出杯时间",
    detail: "删除成本超限、出杯过慢、原料不合法或 SOP 不完整的方案。",
  },
  {
    role: "审核员",
    icon: Search,
    title: "审核员正在搜索市场动态",
    detail: "查看低糖、轻负担、真实茶感和年轻客群的市场信号。",
  },
  {
    role: "审核员",
    icon: Brain,
    title: "审核员综合评价配方",
    detail: "比较市场趋势、商业可行性、产品表达和打样风险。",
  },
  {
    role: "审核员",
    icon: WandSparkles,
    title: "审核员选择最佳方案",
    detail: "从过滤后的候选中选出最适合优先打样的一杯。",
  },
];

const inputClassName =
  "h-11 w-full rounded-xl border border-input bg-white px-3 text-sm font-medium text-foreground shadow-sm outline-none transition focus:ring-2 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-60";

function formatRecipeForClipboard(recipe: DrinkRecipe, index?: number) {
  return [
    `${index ? `候选 ${index}：` : ""}${recipe.name}`,
    "",
    recipe.description,
    "",
    "配料表：",
    ...recipe.ingredients.map((item) => `- ${item.name}：${item.amount}`),
    "",
    "制作流程：",
    ...recipe.steps.map((step, index) => `${index + 1}. ${step}`),
  ].join("\n");
}

function formatResultForClipboard(result: DrinkGenerationResult) {
  return [
    `${result.engineerName}输出的 1 个奶茶候选配方`,
    "",
    `读取 skill：${result.skillReferences.join("、")}`,
    `代码过滤：保留 ${result.filterReport.keptCount} 个，剔除 ${result.filterReport.rejectedCount} 个`,
    `审核员选择：${result.audit.selectedRecipeName}`,
    `审核摘要：${result.audit.summary}`,
    "",
    ...result.recipes.map((recipe, index) =>
      formatRecipeForClipboard(recipe, index + 1),
    ),
  ].join("\n\n---\n\n");
}

export default function Home() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [storeConfig, setStoreConfig] = useState<StoreConfig>(
    getDefaultStoreConfig,
  );
  const [constraints, setConstraints] = useState<GenerationConstraints>({
    season: "夏季",
    targetAudience: "18-30 岁，偏好清爽果香和轻负担口感",
    priceBand: "18-22 元",
    maxIngredientCost: "6 元以内",
    maxMakeTime: "90 秒以内",
    sweetness: "中低甜",
    temperature: "冰饮",
  });
  const [developmentResult, setDevelopmentResult] =
    useState<DrinkDevelopmentResult | null>(null);
  const [auditResult, setAuditResult] = useState<DrinkAuditStageResult | null>(
    null,
  );
  const [status, setStatus] = useState<GenerationStatus>("idle");
  const [recipeStatusIndex, setRecipeStatusIndex] = useState(0);
  const [auditStatusIndex, setAuditStatusIndex] = useState(0);
  const [recipeError, setRecipeError] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [imageError, setImageError] = useState("");
  const [showIngredientTable, setShowIngredientTable] = useState(false);
  const [copied, setCopied] = useState(false);

  const isGeneratingRecipe = status === "generatingDevelopment";
  const isAuditing = status === "auditing";
  const isGeneratingImage = status === "generatingImage";
  const isBusy = isGeneratingRecipe || isAuditing || isGeneratingImage;

  const selectedIngredients = useMemo(
    () =>
      ingredientLibrary.filter((ingredient) =>
        storeConfig.selectedIngredientIds.includes(ingredient.id),
      ),
    [storeConfig.selectedIngredientIds],
  );

  const recipeStatusText = useMemo(
    () =>
      recipeProgressMessages[
        Math.min(recipeStatusIndex, recipeProgressMessages.length - 1)
      ],
    [recipeStatusIndex],
  );
  const activeEngineerStage =
    developmentStages[
      Math.min(recipeStatusIndex, developmentStages.length - 1)
    ];
  const ActiveEngineerIcon = activeEngineerStage.icon;
  const progressPercent =
    ((Math.min(recipeStatusIndex, recipeProgressMessages.length - 1) + 1) /
      recipeProgressMessages.length) *
    100;
  const auditStatusText = useMemo(
    () =>
      auditProgressMessages[
        Math.min(auditStatusIndex, auditProgressMessages.length - 1)
      ],
    [auditStatusIndex],
  );
  const activeAuditStage =
    auditStages[Math.min(auditStatusIndex, auditStages.length - 1)];
  const ActiveAuditIcon = activeAuditStage.icon;
  const auditProgressPercent =
    ((Math.min(auditStatusIndex, auditProgressMessages.length - 1) + 1) /
      auditProgressMessages.length) *
    100;
  const selectedRecipe = useMemo(() => {
    if (!auditResult) {
      return null;
    }

    return (
      auditResult.recipes.find(
        (recipe) => recipe.name === auditResult.audit.selectedRecipeName,
      ) ?? auditResult.recipes[0] ?? null
    );
  }, [auditResult]);
  const finalResult = useMemo<DrinkGenerationResult | null>(() => {
    if (!developmentResult || !auditResult) {
      return null;
    }

    return {
      ...developmentResult,
      ...auditResult,
    };
  }, [auditResult, developmentResult]);

  useEffect(() => {
    function loadStoreConfig() {
      const storedValue = window.localStorage.getItem(storeConfigStorageKey);

      if (!storedValue) {
        setStoreConfig(getDefaultStoreConfig());
        return;
      }

      try {
        setStoreConfig(normalizeStoreConfig(JSON.parse(storedValue)));
      } catch {
        setStoreConfig(getDefaultStoreConfig());
      }
    }

    loadStoreConfig();
    window.addEventListener("focus", loadStoreConfig);

    return () => window.removeEventListener("focus", loadStoreConfig);
  }, []);

  useEffect(() => {
    if (!isGeneratingRecipe) {
      return;
    }

    const timer = window.setInterval(() => {
      setRecipeStatusIndex((current) =>
        Math.min(current + 1, recipeProgressMessages.length - 1),
      );
    }, 900);

    return () => window.clearInterval(timer);
  }, [isGeneratingRecipe]);

  useEffect(() => {
    if (!isAuditing) {
      return;
    }

    const timer = window.setInterval(() => {
      setAuditStatusIndex((current) =>
        Math.min(current + 1, auditProgressMessages.length - 1),
      );
    }, 900);

    return () => window.clearInterval(timer);
  }, [isAuditing]);

  async function handleGenerateRecipe() {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      setRecipeError("先写一点你想要的风味、场景或口感吧。");
      setStatus("recipeError");
      return;
    }

    if (selectedIngredients.length < 4) {
      setRecipeError("至少选择 4 个门店原料，AI 才能组合出完整配方。");
      setStatus("recipeError");
      return;
    }

    setStatus("generatingDevelopment");
    setRecipeStatusIndex(0);
    setAuditStatusIndex(0);
    setRecipeError("");
    setImageUrl("");
    setImageError("");
    setDevelopmentResult(null);
    setAuditResult(null);
    setShowIngredientTable(false);
    setCopied(false);

    try {
      const development = await generateDrinkRecipe({
        prompt: trimmedPrompt,
        storeProfile: storeConfig.storeProfile,
        constraints,
        availableIngredients: selectedIngredients,
      });
      setDevelopmentResult(development);
      setStatus("developmentReady");
      await handleAuditRecipes(development);
    } catch (currentError) {
      setDevelopmentResult(null);
      setAuditResult(null);
      setRecipeError(
        currentError instanceof Error
          ? currentError.message
          : "配方生成失败，请稍后重试。",
      );
      setStatus("recipeError");
    }
  }

  async function handleAuditRecipes(development: DrinkDevelopmentResult) {
    setStatus("auditing");
    setAuditStatusIndex(0);
    setImageUrl("");
    setImageError("");

    try {
      const audit = await auditDrinkRecipes({
        recipes: development.recipes,
        constraints,
        availableIngredients: selectedIngredients,
      });
      setAuditResult(audit);
      setStatus("auditReady");

      const auditedRecipe =
        audit.recipes.find(
          (recipe) => recipe.name === audit.audit.selectedRecipeName,
        ) ?? audit.recipes[0];

      if (auditedRecipe) {
        await handleGenerateImage(auditedRecipe);
      }
    } catch (currentError) {
      setAuditResult(null);
      setRecipeError(
        currentError instanceof Error
          ? currentError.message
          : "审核失败，请稍后重试。",
      );
      setStatus("recipeError");
    }
  }

  async function handleGenerateImage(recipe: DrinkRecipe | null = selectedRecipe) {
    if (!recipe) {
      return;
    }

    setStatus("generatingImage");
    setImageError("");

    try {
      const result = await generateDrinkImage({
        name: recipe.name,
        description: recipe.description,
        ingredients: recipe.ingredients,
      });
      setImageUrl(result.imageUrl);
      setStatus("imageReady");
    } catch (currentError) {
      setImageUrl("");
      setImageError(
        currentError instanceof Error
          ? currentError.message
          : "产品图生成失败，请稍后重试。",
      );
      setStatus("imageError");
    }
  }

  async function handleCopyAll() {
    if (!finalResult) {
      return;
    }

    await navigator.clipboard.writeText(formatResultForClipboard(finalResult));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <main className="min-h-screen bg-[#f2f0eb]">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 lg:px-10">
        <header className="rounded-xl bg-white px-5 py-4 shadow-soft">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
              AI Drink Lab
            </p>
            <h1 className="mt-1 text-2xl font-black text-[#006241] md:text-3xl">
              AI 奶茶配方生成
            </h1>
          </div>
        </header>

        <Card className="p-5 md:p-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-center gap-2 text-[#006241]">
              <Store className="h-5 w-5" />
              <div>
                <h2 className="text-xl font-bold">门店配置</h2>
                <p className="mt-1 text-sm leading-6 text-black/58">
                  当前使用 {storeConfig.storeProfile.storeName} 的门店能力生成配方。
                </p>
              </div>
            </div>
            <Button asChild variant="outline">
              <Link href="/store-config">
                <Settings className="h-4 w-4" />
                配置门店
              </Link>
            </Button>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-border bg-[#fbfaf7] px-4 py-3">
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
                Store
              </p>
              <p className="mt-1 font-bold text-[#1E3932]">
                {storeConfig.storeProfile.storeType}
              </p>
              <p className="mt-1 text-sm text-black/58">
                {storeConfig.storeProfile.brandStyle}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-[#fbfaf7] px-4 py-3">
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
                Equipment
              </p>
              <p className="mt-1 font-bold text-[#1E3932]">
                {storeConfig.storeProfile.equipment.length} 个设备
              </p>
              <p className="mt-1 truncate text-sm text-black/58">
                {storeConfig.storeProfile.equipment.join("、") || "未选择"}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-[#fbfaf7] px-4 py-3">
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
                Ingredients
              </p>
              <p className="mt-1 font-bold text-[#1E3932]">
                {selectedIngredients.length} 个可用原料
              </p>
              <p className="mt-1 truncate text-sm text-black/58">
                {selectedIngredients.map((ingredient) => ingredient.name).join("、")}
              </p>
            </div>
          </div>
        </Card>

        <Card className="overflow-hidden border border-white/80 bg-white">
          <div className="border-b border-border bg-[#fbfaf7] px-5 py-5 md:px-6">
            <div className="flex items-start gap-3 text-[#006241]">
              <div className="engineer-spark mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[#d4e9e2]">
                <Sparkles className="h-5 w-5" />
              </div>
            <div>
              <h2 className="text-xl font-bold">奶茶生成约束</h2>
              <p className="mt-1 text-sm leading-6 text-black/58">
                设置新品研发边界，研发工程师会先读取 skill 库，再输出 1 个候选配方。
              </p>
            </div>
            </div>
          </div>

          <div className="p-5 md:p-6">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              季节
              <select
                value={constraints.season}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    season: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              >
                <option>春季</option>
                <option>夏季</option>
                <option>秋季</option>
                <option>冬季</option>
                <option>全年常规</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              价格带
              <select
                value={constraints.priceBand}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    priceBand: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              >
                <option>12-15 元</option>
                <option>15-18 元</option>
                <option>18-22 元</option>
                <option>22-26 元</option>
                <option>26 元以上</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              成本上限
              <input
                value={constraints.maxIngredientCost}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    maxIngredientCost: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              />
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              出杯时间
              <input
                value={constraints.maxMakeTime}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    maxMakeTime: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              />
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932] md:col-span-2">
              目标人群
              <input
                value={constraints.targetAudience}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    targetAudience: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              />
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              甜度倾向
              <select
                value={constraints.sweetness}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    sweetness: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              >
                <option>低甜</option>
                <option>中低甜</option>
                <option>标准甜</option>
                <option>高甜</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              温度形态
              <select
                value={constraints.temperature}
                onChange={(event) =>
                  setConstraints((current) => ({
                    ...current,
                    temperature: event.target.value,
                  }))
                }
                disabled={isBusy}
                className={inputClassName}
              >
                <option>冰饮</option>
                <option>热饮</option>
                <option>冷热皆可</option>
                <option>少冰</option>
              </select>
            </label>
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-[minmax(0,1fr)_136px] lg:items-end">
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              口味与创意方向
              <Textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="例如：想要一杯适合夏天、清爽、带有青提和茉莉茶香的奶茶"
                disabled={isGeneratingRecipe}
                className="min-h-[104px] resize-none"
              />
            </label>
            <Button
              className="w-full"
              size="default"
              onClick={handleGenerateRecipe}
              disabled={isBusy}
            >
              {isGeneratingRecipe ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
              {isGeneratingRecipe
                ? "研发中"
                : developmentResult
                  ? "重新生成"
                  : "生成 1 个配方"}
            </Button>
          </div>

          {recipeError ? (
            <div className="mt-4 flex gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{recipeError}</span>
            </div>
          ) : null}

          {isGeneratingRecipe ? (
            <div className="engineer-console mt-5 overflow-hidden rounded-2xl border border-[#b7d7cb] bg-[#f6fbf8]">
              <div className="grid gap-0 lg:grid-cols-[minmax(260px,0.8fr)_minmax(0,1.2fr)]">
                <div className="relative overflow-hidden bg-[#1E3932] p-5 text-white">
                  <div className="engineer-scanline" />
                  <div className="relative z-10">
                    <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/60">
                      {activeEngineerStage.role}
                    </p>
                    <div className="mt-5 flex items-center gap-4">
                      <div className="engineer-avatar flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-white text-[#00754A] shadow-soft">
                        <ActiveEngineerIcon className="h-8 w-8" />
                      </div>
                      <div>
                        <h3 className="text-2xl font-black leading-tight">
                          {activeEngineerStage.title}
                        </h3>
                        <p className="mt-2 text-sm leading-6 text-white/70">
                          {activeEngineerStage.detail}
                        </p>
                      </div>
                    </div>
                    <div className="mt-5 flex items-center gap-2 text-sm font-semibold text-white/80">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      {recipeStatusText}
                    </div>
                  </div>
                </div>

                <div className="p-5">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#00754A]">
                        Workflow
                      </p>
                      <h3 className="mt-1 text-lg font-black text-[#1E3932]">
                        研发模块进程
                      </h3>
                    </div>
                    <span className="rounded-full bg-white px-3 py-1 text-xs font-bold text-[#00754A] shadow-soft">
                      {Math.round(progressPercent)}%
                    </span>
                  </div>

                  <div className="h-2 overflow-hidden rounded-full bg-white shadow-inner">
                    <div
                      className="engineer-progress h-full rounded-full bg-[#00754A]"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>

                  <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {developmentStages.map((stage, index) => {
                      const StageIcon = stage.icon;
                      const isActive = index === recipeStatusIndex;
                      const isDone = index < recipeStatusIndex;

                      return (
                        <div
                          key={stage.title}
                          className={`engineer-step rounded-xl border px-3 py-3 transition ${
                            isActive
                              ? "engineer-step-active border-[#00754A] bg-white"
                              : isDone
                                ? "border-[#b7d7cb] bg-white/80"
                                : "border-border bg-[#fbfaf7]"
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <span
                              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                                isActive || isDone
                                  ? "bg-[#00754A] text-white"
                                  : "bg-[#edebe9] text-black/40"
                              }`}
                            >
                              <StageIcon className="h-4 w-4" />
                            </span>
                            <span className="text-sm font-black text-[#1E3932]">
                              {stage.title}
                            </span>
                          </div>
                          <p className="mt-2 text-xs leading-5 text-black/55">
                            {stage.detail}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          ) : null}
          </div>
        </Card>

        {developmentResult ? (
          <section className="space-y-5">
            <Card className="p-5 md:p-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
                    Stage 1 · R&D Module
                  </p>
                  <h2 className="mt-1 text-2xl font-black text-[#006241]">
                    {developmentResult.engineerName}已输出 1 个候选配方
                  </h2>
                  <p className="mt-2 text-sm leading-6 text-black/58">
                    本模块已结束。已读取 skill：
                    {developmentResult.skillReferences.join("、")}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowIngredientTable((current) => !current)}
                >
                  <TableProperties className="h-4 w-4" />
                  {showIngredientTable ? "收起配料表" : "查看配料表"}
                </Button>
              </div>
            </Card>

            {showIngredientTable ? (
              <Card className="overflow-hidden">
                <div className="border-b border-border bg-[#fbfaf7] px-5 py-4">
                  <h3 className="text-lg font-black text-[#006241]">
                    1 个研发候选配方配料表
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[880px] border-collapse text-sm">
                    <thead className="bg-[#f6fbf8] text-left text-[#1E3932]">
                      <tr>
                        <th className="border-b border-border px-4 py-3">候选</th>
                        <th className="border-b border-border px-4 py-3">饮品</th>
                        <th className="border-b border-border px-4 py-3">配料表</th>
                      </tr>
                    </thead>
                    <tbody>
                      {developmentResult.recipes.map((recipe, index) => (
                        <tr key={`${recipe.name}-ingredients`}>
                          <td className="border-b border-border px-4 py-3 font-bold text-[#00754A]">
                            {index + 1}
                          </td>
                          <td className="border-b border-border px-4 py-3 font-bold text-[#1E3932]">
                            {recipe.name}
                          </td>
                          <td className="border-b border-border px-4 py-3 leading-6 text-black/65">
                            {recipe.ingredients
                              .map((item) => `${item.name} ${item.amount}`)
                              .join("；")}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            ) : null}

            {isAuditing ? (
              <div className="engineer-console overflow-hidden rounded-2xl border border-[#b7d7cb] bg-[#f6fbf8]">
                <div className="grid gap-0 lg:grid-cols-[minmax(260px,0.8fr)_minmax(0,1.2fr)]">
                  <div className="relative overflow-hidden bg-[#1E3932] p-5 text-white">
                    <div className="engineer-scanline" />
                    <div className="relative z-10">
                      <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/60">
                        {activeAuditStage.role}
                      </p>
                      <div className="mt-5 flex items-center gap-4">
                        <div className="engineer-avatar flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-white text-[#00754A] shadow-soft">
                          <ActiveAuditIcon className="h-8 w-8" />
                        </div>
                        <div>
                          <h3 className="text-2xl font-black leading-tight">
                            {activeAuditStage.title}
                          </h3>
                          <p className="mt-2 text-sm leading-6 text-white/70">
                            {activeAuditStage.detail}
                          </p>
                        </div>
                      </div>
                      <div className="mt-5 flex items-center gap-2 text-sm font-semibold text-white/80">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        {auditStatusText}
                      </div>
                    </div>
                  </div>

                  <div className="p-5">
                    <div className="mb-4 flex items-center justify-between gap-3">
                      <div>
                        <p className="text-xs font-bold uppercase tracking-[0.14em] text-[#00754A]">
                          Stage 2
                        </p>
                        <h3 className="mt-1 text-lg font-black text-[#1E3932]">
                          审核模块进程
                        </h3>
                      </div>
                      <span className="rounded-full bg-white px-3 py-1 text-xs font-bold text-[#00754A] shadow-soft">
                        {Math.round(auditProgressPercent)}%
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-white shadow-inner">
                      <div
                        className="engineer-progress h-full rounded-full bg-[#00754A]"
                        style={{ width: `${auditProgressPercent}%` }}
                      />
                    </div>

                    <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                      {auditStages.map((stage, index) => {
                        const StageIcon = stage.icon;
                        const isActive = index === auditStatusIndex;
                        const isDone = index < auditStatusIndex;

                        return (
                          <div
                            key={stage.title}
                            className={`engineer-step rounded-xl border px-3 py-3 transition ${
                              isActive
                                ? "engineer-step-active border-[#00754A] bg-white"
                                : isDone
                                  ? "border-[#b7d7cb] bg-white/80"
                                  : "border-border bg-[#fbfaf7]"
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <span
                                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                                  isActive || isDone
                                    ? "bg-[#00754A] text-white"
                                    : "bg-[#edebe9] text-black/40"
                                }`}
                              >
                                <StageIcon className="h-4 w-4" />
                              </span>
                              <span className="text-sm font-black text-[#1E3932]">
                                {stage.title}
                              </span>
                            </div>
                            <p className="mt-2 text-xs leading-5 text-black/55">
                              {stage.detail}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            {auditResult ? (
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
              <Card className="overflow-hidden">
                <div className="bg-[#00754A] px-5 py-5 text-white md:px-6">
                  <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/70">
                    Stage 2 · Auditor Pick
                  </p>
                  <h2 className="mt-1 text-2xl font-black">
                    {auditResult.audit.auditorName}选出：
                    {auditResult.audit.selectedRecipeName}
                  </h2>
                  <p className="mt-3 max-w-4xl text-sm leading-7 text-white/80">
                    {auditResult.audit.summary}
                  </p>
                </div>
                <div className="grid gap-4 p-5 md:grid-cols-[minmax(0,1fr)_320px] md:p-6">
                  <div>
                    <h3 className="text-lg font-bold text-[#006241]">
                      选择理由
                    </h3>
                    <div className="mt-3 grid gap-2">
                      {auditResult.audit.reasons.map((reason) => (
                        <div
                          key={reason}
                          className="rounded-xl border border-border bg-[#fbfaf7] px-4 py-3 text-sm leading-6 text-black/70"
                        >
                          {reason}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-[#006241]">
                      代码过滤
                    </h3>
                    <div className="mt-3 rounded-xl border border-border bg-[#fbfaf7] px-4 py-3">
                      <p className="text-sm font-bold text-[#1E3932]">
                        保留 {auditResult.filterReport.keptCount} 个，剔除{" "}
                        {auditResult.filterReport.rejectedCount} 个
                      </p>
                      <p className="mt-2 text-xs leading-5 text-black/58">
                        过滤依据：原料合法性、成本上限、出杯时间、配料数量和制作步骤完整度。
                      </p>
                    </div>
                    <h3 className="mt-4 text-lg font-bold text-[#006241]">
                      市场信号
                    </h3>
                    <div className="mt-3 grid gap-2">
                      {auditResult.audit.marketSignals.map((signal) => (
                        <div
                          key={signal}
                          className="rounded-xl bg-[#d4e9e2] px-3 py-2 text-xs font-semibold leading-5 text-[#1E3932]"
                        >
                          {signal}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>

              <DrinkImageCard
                recipe={selectedRecipe}
                imageUrl={imageUrl}
                status={status}
                error={imageError}
                onRegenerateImage={() => handleGenerateImage()}
              />
            </div>
            ) : null}

            {finalResult ? (
              <div className="flex justify-end">
                <Button type="button" variant="outline" onClick={handleCopyAll}>
                  <Clipboard className="h-4 w-4" />
                  {copied ? "已复制全部" : "复制全部"}
                </Button>
              </div>
            ) : null}

            <div className="grid gap-5 xl:grid-cols-2">
              {developmentResult.recipes.map((currentRecipe, index) => (
                <RecipeCard
                  key={`${currentRecipe.name}-${index}`}
                  recipe={currentRecipe}
                  titlePrefix={`候选 ${index + 1}`}
                  copied={copied}
                  onCopy={async () => {
                    await navigator.clipboard.writeText(
                      formatRecipeForClipboard(currentRecipe, index + 1),
                    );
                    setCopied(true);
                    window.setTimeout(() => setCopied(false), 1600);
                  }}
                  onRegenerateRecipe={handleGenerateRecipe}
                  isGeneratingRecipe={isGeneratingRecipe}
                />
              ))}
            </div>
          </section>
        ) : (
          <Card className="grid min-h-[440px] place-items-center overflow-hidden bg-white">
            <div className="max-w-md px-8 text-center">
              <div className="mx-auto mb-6 flex h-24 w-24 items-center justify-center rounded-full bg-[#d4e9e2] text-[#006241]">
                <Sparkles className="h-10 w-10" />
              </div>
              <h2 className="text-3xl font-black text-[#006241]">
                等待第一杯灵感
              </h2>
              <p className="mt-4 text-base leading-8 text-black/58">
                输入风味、季节、目标人群或出品场景。研发工程师会先读取 skill 库，再输出 1 个奶茶候选方案。
              </p>
            </div>
          </Card>
        )}
      </section>
    </main>
  );
}
