"use client";

import {
  Check,
  Clipboard,
  ImageOff,
  Loader2,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { DrinkRecipe, GenerationStatus } from "@/types/drink";

type RecipeCardProps = {
  recipe: DrinkRecipe;
  titlePrefix?: string;
  copied: boolean;
  onCopy: () => void;
  onRegenerateRecipe: () => void;
  isGeneratingRecipe: boolean;
};

type DrinkImageCardProps = {
  recipe: DrinkRecipe | null;
  imageUrl: string;
  status: GenerationStatus;
  error: string;
  onRegenerateImage: () => void;
};

export function RecipeCard({
  recipe,
  titlePrefix,
  copied,
  onCopy,
  onRegenerateRecipe,
  isGeneratingRecipe,
}: RecipeCardProps) {
  return (
    <Card className="overflow-hidden">
      <div className="bg-[#1E3932] px-6 py-6 text-white md:px-8">
        {titlePrefix ? (
          <p className="mb-2 text-xs font-bold uppercase tracking-[0.12em] text-white/60">
            {titlePrefix}
          </p>
        ) : null}
        <h2 className="text-3xl font-black leading-tight md:text-4xl">
          {recipe.name}
        </h2>
        <p className="mt-4 max-w-3xl text-base leading-8 text-white/75">
          {recipe.description}
        </p>
      </div>

      <div className="space-y-7 p-6 md:p-8">
        <div>
          <div className="mb-4 flex items-center justify-between gap-3">
            <h3 className="text-xl font-bold text-[#006241]">配料表</h3>
            <span className="rounded-full bg-[#d4e9e2] px-3 py-1 text-xs font-semibold text-[#1E3932]">
              单杯建议克重
            </span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {recipe.ingredients.map((ingredient, index) => (
              <div
                key={ingredient.name}
                className="flex items-center justify-between gap-4 rounded-xl border border-border bg-white px-4 py-3"
              >
                <span className="font-medium text-foreground">
                  {index + 1}. {ingredient.name}
                </span>
                <span className="text-lg font-black text-[#00754A]">
                  {ingredient.amount}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="mb-4 text-xl font-bold text-[#006241]">制作流程</h3>
          <ol className="grid gap-3">
            {recipe.steps.map((step, index) => (
              <li key={step} className="flex gap-3 rounded-xl bg-[#f7f5f0] p-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#00754A] text-sm font-bold text-white">
                  {index + 1}
                </span>
                <span className="pt-1 text-base leading-7 text-black/70">
                  {step}
                </span>
              </li>
            ))}
          </ol>
        </div>

        <div className="flex flex-wrap gap-3 pt-1">
          <Button onClick={onCopy} variant="outline">
            {copied ? (
              <Check className="h-4 w-4" />
            ) : (
              <Clipboard className="h-4 w-4" />
            )}
            {copied ? "已复制" : "复制配方"}
          </Button>
          <Button onClick={onRegenerateRecipe} disabled={isGeneratingRecipe}>
            <Sparkles className="h-4 w-4" />
            重新生成 1 个
          </Button>
        </div>
      </div>
    </Card>
  );
}

export function DrinkImageCard({
  recipe,
  imageUrl,
  status,
  error,
  onRegenerateImage,
}: DrinkImageCardProps) {
  const isGenerating = status === "generatingImage";
  const isError = status === "imageError";
  const isReady = status === "imageReady" && imageUrl;

  return (
    <Card className="h-fit overflow-hidden bg-white">
      <div className="p-5">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
              Preview
            </p>
            <h3 className="mt-1 text-xl font-black text-[#006241]">
              奶茶产品图
            </h3>
          </div>
          {recipe ? (
            <Button
              size="sm"
              variant="outline"
              onClick={onRegenerateImage}
              disabled={isGenerating}
            >
              {isGenerating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              重生成图片
            </Button>
          ) : null}
        </div>

        <div className="relative aspect-[4/5] overflow-hidden rounded-xl bg-[#edebe9]">
          {isReady ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={imageUrl}
              alt={`${recipe?.name ?? "奶茶"} 产品图`}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="absolute inset-x-8 bottom-8 top-10 rounded-[42px] border border-white/70 bg-white/45 shadow-float backdrop-blur-sm">
              <div className="product-photo absolute inset-x-7 bottom-7 top-10 overflow-hidden rounded-[34px] border border-white/80">
                <div className="glass-shine absolute inset-y-0 left-6 w-10 rotate-6 rounded-full blur-[1px]" />
                <div className="absolute inset-x-6 top-7 h-12 rounded-full bg-white/85 shadow-sm" />
                <div className="absolute inset-x-10 top-11 h-8 rounded-full bg-[#f8fff7]" />
                <div className="absolute bottom-8 left-1/2 h-20 w-28 -translate-x-1/2 rounded-full bg-[#b9dc88]/70 blur-xl" />
              </div>
              <div className="absolute left-1/2 top-2 h-12 w-28 -translate-x-1/2 rounded-full bg-white shadow-soft" />
              <div className="absolute left-1/2 top-5 h-4 w-40 -translate-x-1/2 rounded-full bg-white/85" />
            </div>
          )}

          {isGenerating ? (
            <div className="absolute inset-0 grid place-items-center bg-white/75 p-6 text-center backdrop-blur-sm">
              <div>
                <Loader2 className="mx-auto mb-4 h-8 w-8 animate-spin text-[#00754A]" />
                <p className="text-lg font-black text-[#006241]">
                  正在生成产品图
                </p>
                <p className="mt-2 text-sm leading-6 text-black/58">
                  已收到饮品方案，正在根据名称、描述和配料生成预览图。
                </p>
              </div>
            </div>
          ) : null}

          {isError ? (
            <div className="absolute inset-0 grid place-items-center bg-white/86 p-6 text-center backdrop-blur-sm">
              <div>
                <ImageOff className="mx-auto mb-4 h-9 w-9 text-red-600" />
                <p className="text-lg font-black text-[#1E3932]">
                  图片生成失败
                </p>
                <p className="mt-2 text-sm leading-6 text-black/58">
                  {error || "配方已生成，可重新生成图片。"}
                </p>
                <Button className="mt-5" onClick={onRegenerateImage}>
                  <RefreshCw className="h-4 w-4" />
                  重新生成图片
                </Button>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
