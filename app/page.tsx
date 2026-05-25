"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  AlertCircle,
  BookOpen,
  Brain,
  ChartColumn,
  Clipboard,
  FlaskConical,
  LibraryBig,
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
  rateDrinkRecipe,
} from "@/lib/api";
import {
  getAllIngredients,
  getDefaultStoreConfig,
  normalizeStoreConfig,
  storeConfigStorageKey,
} from "@/lib/store-config";
import type { StoreConfig } from "@/lib/store-config";
import type {
  DrinkRecipe,
  DrinkFeedback,
  DrinkAuditStageResult,
  DrinkDevelopmentResult,
  DrinkGenerationResult,
  GenerationConstraints,
  GenerationStatus,
  RateDrinkResult,
} from "@/types/drink";

const defaultPrompt =
  "我想要一杯适合夏天、清爽、带有青提和茉莉茶香的奶茶";

const recipeProgressMessages = [
  "正在阅读 skill",
  "搜索信息中",
  "绞尽脑汁中",
  "校验门店约束",
  "调配候选配方",
  "输出候选方案",
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
    detail: "组合差异化方案，避免重复和不可执行配方。",
  },
  {
    role: "研发工程师",
    icon: WandSparkles,
    title: "输出候选方案",
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
const homeSessionStorageKey = "ai-drink-lab-home-session";

type HomeSession = {
  prompt?: string;
  constraints?: GenerationConstraints;
  generationCount?: number;
  developmentResult?: DrinkDevelopmentResult | null;
  auditResult?: DrinkAuditStageResult | null;
  status?: GenerationStatus;
  imageUrl?: string;
  imageError?: string;
  showIngredientTable?: boolean;
  feedbacks?: DrinkFeedback[];
  feedbackDraft?: DrinkFeedback;
  ratingStatus?: "idle" | "saving" | "saved" | "error";
  ratingMessage?: string;
  ratingResult?: RateDrinkResult | null;
};

function randomStageDelay() {
  return 3000 + Math.floor(Math.random() * 2001);
}

function restoreStableStatus(session: HomeSession): GenerationStatus {
  if (session.status === "imageError") {
    return "imageError";
  }
  if (session.imageUrl) {
    return "imageReady";
  }
  if (session.auditResult) {
    return "auditReady";
  }
  if (session.developmentResult) {
    return "developmentReady";
  }
  return "idle";
}

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
    `${result.engineerName}输出的 ${result.recipes.length} 个奶茶候选配方`,
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

type FeedbackPanelProps = {
  recipe: DrinkRecipe | null;
  feedbacks: DrinkFeedback[];
  draft: DrinkFeedback;
  averageRating: number;
  status: "idle" | "saving" | "saved" | "error";
  result: RateDrinkResult | null;
  message: string;
  onDraftChange: (draft: DrinkFeedback) => void;
  onAddFeedback: () => void;
  onSubmit: () => void;
};

function FeedbackPanel({
  recipe,
  feedbacks,
  draft,
  averageRating,
  status,
  result,
  message,
  onDraftChange,
  onAddFeedback,
  onSubmit,
}: FeedbackPanelProps) {
  const isSaving = status === "saving";
  const hasFeedback = feedbacks.length > 0;
  const targetLibrary = averageRating > 7 ? "好配方库" : "差配方库";

  return (
    <Card className="overflow-hidden">
      <div className="border-b border-border bg-[#fbfaf7] px-5 py-4">
        <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
          用户反馈
        </p>
        <h3 className="mt-1 text-xl font-black text-[#006241]">
          反馈收集
        </h3>
      </div>
      <div className="p-5">
        <div className="rounded-xl bg-[#f6fbf8] px-4 py-3">
          <p className="text-sm font-semibold text-black/58">当前最终配方</p>
          <p className="mt-1 text-lg font-black text-[#1E3932]">
            {recipe?.name ?? "等待审核结果"}
          </p>
        </div>

        <div className="mt-4 rounded-xl border border-border bg-white p-3">
          <div className="grid gap-3 sm:grid-cols-[96px_minmax(0,1fr)]">
            <label className="grid gap-1 text-xs font-bold text-[#1E3932]">
              分数
              <input
                type="number"
                min={0}
                max={10}
                step={1}
                value={draft.score}
                onChange={(event) =>
                  onDraftChange({
                    ...draft,
                    score: Math.max(
                      0,
                      Math.min(10, Number(event.target.value) || 0),
                    ),
                  })
                }
                disabled={!recipe || isSaving}
                className="h-10 rounded-lg border border-input bg-white px-2 text-center text-sm font-black outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
              />
            </label>
            <label className="grid gap-1 text-xs font-bold text-[#1E3932]">
              评价
              <input
                value={draft.comment}
                onChange={(event) =>
                  onDraftChange({ ...draft, comment: event.target.value })
                }
                disabled={!recipe || isSaving}
                placeholder="例如：青提香气自然，尾段茶涩略重"
                className="h-10 rounded-lg border border-input bg-white px-3 text-sm font-medium outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
              />
            </label>
          </div>
          <Button
            type="button"
            variant="outline"
            className="mt-3 w-full"
            onClick={onAddFeedback}
            disabled={!recipe || isSaving}
          >
            新增反馈
          </Button>
        </div>

        <div className="mt-4 grid gap-2">
          {feedbacks.map((feedback, index) => (
            <div
              key={`${feedback.createdAt ?? "draft"}-${index}`}
              className="rounded-xl border border-border bg-[#fbfaf7] px-3 py-2"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs font-bold text-black/45">
                  反馈 {index + 1}
                </span>
                <span className="text-lg font-black text-[#00754A]">
                  {feedback.score}
                </span>
              </div>
              <p className="mt-1 text-sm leading-6 text-black/65">
                {feedback.comment || "未填写文字评价"}
              </p>
            </div>
          ))}

          {!feedbacks.length ? (
            <div className="rounded-xl border border-dashed border-border bg-[#fbfaf7] px-3 py-4 text-center text-sm font-semibold text-black/45">
              暂无反馈，先新增一条分数和评价。
            </div>
          ) : null}
        </div>

        <div className="mt-4 rounded-xl border border-border bg-white px-4 py-3">
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-bold text-black/58">平均分</span>
            <span className="text-3xl font-black text-[#00754A]">
              {averageRating.toFixed(1)}
            </span>
          </div>
          <p className="mt-2 text-xs font-semibold leading-5 text-black/50">
            已收集 {feedbacks.length} 条反馈。平均分大于 7 存入好配方库，其余存入差配方库。
            {hasFeedback ? ` 当前将进入：${targetLibrary}` : ""}
          </p>
        </div>

        <Button
          type="button"
          className="mt-4 w-full"
          onClick={onSubmit}
          disabled={!recipe || isSaving || !hasFeedback}
        >
          {isSaving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
          {isSaving ? "保存反馈中" : "提交反馈并入库"}
        </Button>

        {message ? (
          <p
            className={`mt-3 rounded-xl px-3 py-2 text-sm font-semibold leading-6 ${
              status === "error"
                ? "bg-red-50 text-red-700"
                : "bg-[#d4e9e2] text-[#1E3932]"
            }`}
          >
            {message}
          </p>
        ) : null}

        {result ? (
          <p className="mt-2 text-xs leading-5 text-black/45">
            入库 ID：{result.id}
          </p>
        ) : null}
      </div>
    </Card>
  );
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
  const [generationCount, setGenerationCount] = useState(3);
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
  const [feedbacks, setFeedbacks] = useState<DrinkFeedback[]>([]);
  const [feedbackDraft, setFeedbackDraft] = useState<DrinkFeedback>({
    score: 8,
    comment: "",
  });
  const [ratingStatus, setRatingStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const [ratingMessage, setRatingMessage] = useState("");
  const [ratingResult, setRatingResult] = useState<RateDrinkResult | null>(null);
  const [hasLoadedSession, setHasLoadedSession] = useState(false);

  const isGeneratingRecipe = status === "generatingDevelopment";
  const isAuditing = status === "auditing";
  const isGeneratingImage = status === "generatingImage";
  const isBusy = isGeneratingRecipe || isAuditing || isGeneratingImage;

  const selectedIngredients = useMemo(
    () =>
      getAllIngredients(storeConfig).filter((ingredient) =>
        storeConfig.selectedIngredientIds.includes(ingredient.id),
      ),
    [storeConfig],
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
  const averageRating = useMemo(
    () =>
      feedbacks.length
        ? feedbacks.reduce((sum, feedback) => sum + feedback.score, 0) /
          feedbacks.length
        : 0,
    [feedbacks],
  );

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
    const sessionValue = window.localStorage.getItem(homeSessionStorageKey);

    if (sessionValue) {
      try {
        const session = JSON.parse(sessionValue) as HomeSession;

        if (typeof session.prompt === "string") {
          setPrompt(session.prompt);
        }
        if (session.constraints) {
          setConstraints(session.constraints);
        }
        if (typeof session.generationCount === "number") {
          setGenerationCount(
            Math.max(1, Math.min(6, Math.round(session.generationCount))),
          );
        }
        if (session.developmentResult) {
          setDevelopmentResult(session.developmentResult);
        }
        if (session.auditResult) {
          setAuditResult(session.auditResult);
        }
        setStatus(restoreStableStatus(session));
        setRecipeStatusIndex(recipeProgressMessages.length - 1);
        setAuditStatusIndex(auditProgressMessages.length - 1);
        setImageUrl(session.imageUrl ?? "");
        setImageError(session.imageError ?? "");
        setShowIngredientTable(Boolean(session.showIngredientTable));
        if (Array.isArray(session.feedbacks)) {
          setFeedbacks(
            session.feedbacks.map((feedback) => ({
              score: Math.max(0, Math.min(10, Number(feedback.score) || 0)),
              comment: String(feedback.comment ?? ""),
              createdAt: feedback.createdAt,
            })),
          );
        }
        if (session.feedbackDraft) {
          setFeedbackDraft({
            score: Math.max(
              0,
              Math.min(10, Number(session.feedbackDraft.score) || 0),
            ),
            comment: String(session.feedbackDraft.comment ?? ""),
          });
        }
        if (session.ratingStatus && session.ratingStatus !== "saving") {
          setRatingStatus(session.ratingStatus);
        }
        setRatingMessage(session.ratingMessage ?? "");
        setRatingResult(session.ratingResult ?? null);
      } catch {
        window.localStorage.removeItem(homeSessionStorageKey);
      }
    }

    setHasLoadedSession(true);
    window.addEventListener("focus", loadStoreConfig);

    return () => window.removeEventListener("focus", loadStoreConfig);
  }, []);

  useEffect(() => {
    if (!hasLoadedSession) {
      return;
    }

    const session: HomeSession = {
      prompt,
      constraints,
      generationCount,
      developmentResult,
      auditResult,
      status:
        isGeneratingRecipe || isAuditing || isGeneratingImage
          ? restoreStableStatus({ developmentResult, auditResult, imageUrl })
          : status,
      imageUrl,
      imageError,
      showIngredientTable,
      feedbacks,
      feedbackDraft,
      ratingStatus: ratingStatus === "saving" ? "idle" : ratingStatus,
      ratingMessage,
      ratingResult,
    };

    window.localStorage.setItem(homeSessionStorageKey, JSON.stringify(session));
  }, [
    auditResult,
    constraints,
    developmentResult,
    feedbackDraft,
    feedbacks,
    generationCount,
    hasLoadedSession,
    imageError,
    imageUrl,
    isAuditing,
    isGeneratingImage,
    isGeneratingRecipe,
    prompt,
    ratingMessage,
    ratingResult,
    ratingStatus,
    showIngredientTable,
    status,
  ]);

  useEffect(() => {
    if (!isGeneratingRecipe) {
      return;
    }

    let timer: number | undefined;

    function scheduleNextStage() {
      timer = window.setTimeout(() => {
        setRecipeStatusIndex((current) => {
          const next = Math.min(current + 1, recipeProgressMessages.length - 1);

          if (next < recipeProgressMessages.length - 1) {
            scheduleNextStage();
          }

          return next;
        });
      }, randomStageDelay());
    }

    scheduleNextStage();

    return () => {
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [isGeneratingRecipe]);

  useEffect(() => {
    if (!isAuditing) {
      return;
    }

    let timer: number | undefined;

    function scheduleNextStage() {
      timer = window.setTimeout(() => {
        setAuditStatusIndex((current) => {
          const next = Math.min(current + 1, auditProgressMessages.length - 1);

          if (next < auditProgressMessages.length - 1) {
            scheduleNextStage();
          }

          return next;
        });
      }, randomStageDelay());
    }

    scheduleNextStage();

    return () => {
      if (timer) {
        window.clearTimeout(timer);
      }
    };
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
    setFeedbacks([]);
    setFeedbackDraft({ score: 8, comment: "" });
    setRatingStatus("idle");
    setRatingMessage("");
    setRatingResult(null);

    try {
      const development = await generateDrinkRecipe({
        prompt: trimmedPrompt,
        generationCount,
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

  function handleAddFeedback() {
    setRatingStatus("idle");
    setRatingMessage("");
    setRatingResult(null);
    setFeedbacks((current) => [
      ...current,
      {
        score: Math.max(0, Math.min(10, Number(feedbackDraft.score) || 0)),
        comment: feedbackDraft.comment.trim(),
        createdAt: new Date().toISOString(),
      },
    ]);
    setFeedbackDraft({ score: 8, comment: "" });
  }

  async function handleSubmitRating() {
    if (!selectedRecipe || !feedbacks.length) {
      return;
    }

    setRatingStatus("saving");
    setRatingMessage("");
    setRatingResult(null);

    try {
      const result = await rateDrinkRecipe({
        recipe: selectedRecipe,
        feedbacks,
        constraints,
      });
      setRatingResult(result);
      setRatingStatus("saved");
      setRatingMessage(
        `平均分 ${result.averageRating.toFixed(1)}，已保存到 ${
          result.library === "recipe_skill_library" ? "好配方库" : "差配方库"
        }。`,
      );
    } catch (currentError) {
      setRatingStatus("error");
      setRatingMessage(
        currentError instanceof Error
          ? currentError.message
          : "反馈保存失败，请稍后重试。",
      );
    }
  }

  return (
    <main className="min-h-screen bg-[#f2f0eb]">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 lg:px-10">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-xl bg-white px-5 py-4 shadow-soft">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
              AI 奶茶实验室
            </p>
            <h1 className="mt-1 text-2xl font-black text-[#006241] md:text-3xl">
              AI 奶茶配方生成
            </h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline">
              <Link href="/history">
                <ChartColumn className="h-4 w-4" />
                历史回溯
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/skill-library">
                <LibraryBig className="h-4 w-4" />
                Skill 库
              </Link>
            </Button>
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
                门店
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
                设备
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
                原料
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
                设置新品研发边界，研发工程师会先读取 skill 库，再输出 K 个候选配方。
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
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              生成数量 K
              <input
                type="number"
                min={1}
                max={6}
                step={1}
                value={generationCount}
                onChange={(event) =>
                  setGenerationCount(
                    Math.max(1, Math.min(6, Number(event.target.value) || 1)),
                  )
                }
                disabled={isBusy}
                className={inputClassName}
              />
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
                  : `生成 ${generationCount} 个配方`}
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
                        工作流
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
                    阶段 1 · 研发模块
                  </p>
                  <h2 className="mt-1 text-2xl font-black text-[#006241]">
                    {developmentResult.engineerName}已输出{" "}
                    {developmentResult.recipes.length} 个候选配方
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
                    {developmentResult.recipes.length} 个研发候选配方配料表
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
                          阶段 2
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
            <>
              <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
                <Card className="overflow-hidden">
                  <div className="bg-[#00754A] px-5 py-5 text-white md:px-6">
                    <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/70">
                      阶段 2 · 审核挑选
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
              <FeedbackPanel
                recipe={selectedRecipe}
                feedbacks={feedbacks}
                draft={feedbackDraft}
                averageRating={averageRating}
                status={ratingStatus}
                result={ratingResult}
                message={ratingMessage}
                onDraftChange={setFeedbackDraft}
                onAddFeedback={handleAddFeedback}
                onSubmit={handleSubmitRating}
              />
            </>
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
                输入风味、季节、目标人群或出品场景。研发工程师会先读取 skill 库，再输出你设置数量的奶茶候选方案。
              </p>
            </div>
          </Card>
        )}
      </section>
    </main>
  );
}
