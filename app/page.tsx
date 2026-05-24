"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Loader2, Send, Sparkles } from "lucide-react";
import { DrinkImageCard, RecipeCard } from "@/components/recipe-card";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { generateDrinkImage, generateDrinkRecipe } from "@/lib/api";
import type { DrinkRecipe, GenerationStatus } from "@/types/drink";

const defaultPrompt =
  "我想要一杯适合夏天、清爽、带有青提和茉莉茶香的奶茶";

const recipeProgressMessages = [
  "Agent 正在分析需求",
  "正在拆解风味关键词",
  "正在设计配方比例",
  "正在输出饮品方案",
];

function formatRecipeForClipboard(recipe: DrinkRecipe) {
  return [
    recipe.name,
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

export default function Home() {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [recipe, setRecipe] = useState<DrinkRecipe | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [status, setStatus] = useState<GenerationStatus>("idle");
  const [recipeStatusIndex, setRecipeStatusIndex] = useState(0);
  const [recipeError, setRecipeError] = useState("");
  const [imageError, setImageError] = useState("");
  const [copied, setCopied] = useState(false);

  const isGeneratingRecipe = status === "generatingRecipe";
  const isGeneratingImage = status === "generatingImage";
  const isBusy = isGeneratingRecipe || isGeneratingImage;

  const recipeStatusText = useMemo(
    () =>
      recipeProgressMessages[
        Math.min(recipeStatusIndex, recipeProgressMessages.length - 1)
      ],
    [recipeStatusIndex],
  );

  useEffect(() => {
    if (!isGeneratingRecipe) {
      return;
    }

    const timer = window.setInterval(() => {
      setRecipeStatusIndex((current) =>
        Math.min(current + 1, recipeProgressMessages.length - 1),
      );
    }, 650);

    return () => window.clearInterval(timer);
  }, [isGeneratingRecipe]);

  async function requestDrinkImage(nextRecipe: DrinkRecipe) {
    setStatus("generatingImage");
    setImageError("");
    setImageUrl("");

    try {
      const result = await generateDrinkImage({
        name: nextRecipe.name,
        description: nextRecipe.description,
        ingredients: nextRecipe.ingredients,
      });
      setImageUrl(result.imageUrl);
      setStatus("imageReady");
    } catch (currentError) {
      setImageError(
        currentError instanceof Error
          ? currentError.message
          : "图片生成失败，可重新生成图片。",
      );
      setStatus("imageError");
    }
  }

  async function handleGenerateRecipe() {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      setRecipeError("先写一点你想要的风味、场景或口感吧。");
      setStatus("recipeError");
      return;
    }

    setStatus("generatingRecipe");
    setRecipeStatusIndex(0);
    setRecipeError("");
    setImageError("");
    setImageUrl("");
    setCopied(false);

    try {
      const result = await generateDrinkRecipe({ prompt: trimmedPrompt });
      setRecipe(result);
      setStatus("recipeReady");
      await requestDrinkImage(result);
    } catch (currentError) {
      setRecipe(null);
      setRecipeError(
        currentError instanceof Error
          ? currentError.message
          : "配方生成失败，请稍后重试。",
      );
      setStatus("recipeError");
    }
  }

  async function handleRegenerateImage() {
    if (!recipe) {
      return;
    }

    await requestDrinkImage(recipe);
  }

  async function handleCopy() {
    if (!recipe) {
      return;
    }

    await navigator.clipboard.writeText(formatRecipeForClipboard(recipe));
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
          <div className="mb-4 flex items-center gap-2 text-[#006241]">
            <Sparkles className="h-5 w-5" />
            <h2 className="text-xl font-bold">描述你想要的饮品</h2>
          </div>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_136px] lg:items-center">
            <Textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="例如：想要一杯适合夏天、清爽、带有青提和茉莉茶香的奶茶"
              disabled={isGeneratingRecipe}
              className="min-h-[104px] resize-none"
            />
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
                ? "生成配方中"
                : recipe
                  ? "重新生成"
                  : "生成配方"}
            </Button>
          </div>

          {recipeError ? (
            <div className="mt-4 flex gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{recipeError}</span>
            </div>
          ) : null}

          {isGeneratingRecipe ? (
            <div className="mt-4 rounded-xl bg-[#d4e9e2] px-4 py-3 text-sm font-semibold text-[#1E3932]">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                {recipeStatusText}
              </div>
              <div className="mt-3 grid grid-cols-4 gap-2">
                {recipeProgressMessages.map((message, index) => (
                  <div
                    key={message}
                    className={`h-1.5 rounded-full ${
                      index <= recipeStatusIndex ? "bg-[#00754A]" : "bg-white"
                    }`}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </Card>

        {recipe ? (
          <section className="grid gap-5 lg:grid-cols-[minmax(0,2.4fr)_minmax(286px,0.9fr)]">
            <RecipeCard
              recipe={recipe}
              copied={copied}
              onCopy={handleCopy}
              onRegenerateRecipe={handleGenerateRecipe}
              isGeneratingRecipe={isGeneratingRecipe}
            />
            <DrinkImageCard
              recipe={recipe}
              imageUrl={imageUrl}
              status={status}
              error={imageError}
              onRegenerateImage={handleRegenerateImage}
            />
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
                输入风味、季节、目标人群或出品场景。页面会先生成饮品方案，再独立生成右侧产品图。
              </p>
            </div>
          </Card>
        )}
      </section>
    </main>
  );
}
