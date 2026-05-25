"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Check,
  Database,
  Save,
  SlidersHorizontal,
  Store,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  defaultSelectedIngredientIds,
  equipmentOptions,
  getDefaultStoreConfig,
  ingredientLibrary,
  normalizeStoreConfig,
  storeConfigStorageKey,
} from "@/lib/store-config";
import type { StoreConfig } from "@/lib/store-config";
import type { StoreIngredient } from "@/types/drink";

const inputClassName =
  "h-11 w-full rounded-xl border border-input bg-white px-3 text-sm font-medium text-foreground shadow-sm outline-none transition focus:ring-2 focus:ring-ring";

export default function StoreConfigPage() {
  const router = useRouter();
  const [storeConfig, setStoreConfig] = useState<StoreConfig>(
    getDefaultStoreConfig,
  );
  const [saved, setSaved] = useState(false);

  const selectedIngredients = useMemo(
    () =>
      ingredientLibrary.filter((ingredient) =>
        storeConfig.selectedIngredientIds.includes(ingredient.id),
      ),
    [storeConfig.selectedIngredientIds],
  );

  const ingredientGroups = useMemo(
    () =>
      ingredientLibrary.reduce(
        (groups, ingredient) => {
          groups[ingredient.category] = [
            ...(groups[ingredient.category] ?? []),
            ingredient,
          ];
          return groups;
        },
        {} as Record<StoreIngredient["category"], StoreIngredient[]>,
      ),
    [],
  );

  useEffect(() => {
    const storedValue = window.localStorage.getItem(storeConfigStorageKey);

    if (!storedValue) {
      return;
    }

    try {
      setStoreConfig(normalizeStoreConfig(JSON.parse(storedValue)));
    } catch {
      setStoreConfig(getDefaultStoreConfig());
    }
  }, []);

  function toggleEquipment(equipment: string) {
    setSaved(false);
    setStoreConfig((current) => ({
      ...current,
      storeProfile: {
        ...current.storeProfile,
        equipment: current.storeProfile.equipment.includes(equipment)
          ? current.storeProfile.equipment.filter((item) => item !== equipment)
          : [...current.storeProfile.equipment, equipment],
      },
    }));
  }

  function toggleIngredient(ingredientId: string) {
    setSaved(false);
    setStoreConfig((current) => ({
      ...current,
      selectedIngredientIds: current.selectedIngredientIds.includes(ingredientId)
        ? current.selectedIngredientIds.filter((id) => id !== ingredientId)
        : [...current.selectedIngredientIds, ingredientId],
    }));
  }

  function saveConfig() {
    window.localStorage.setItem(
      storeConfigStorageKey,
      JSON.stringify(storeConfig),
    );
    setSaved(true);
  }

  function saveAndReturn() {
    saveConfig();
    router.push("/");
  }

  return (
    <main className="min-h-screen bg-[#f2f0eb]">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 md:px-8 lg:px-10">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-xl bg-white px-5 py-4 shadow-soft">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
              Store Setup
            </p>
            <h1 className="mt-1 text-2xl font-black text-[#006241] md:text-3xl">
              门店配置
            </h1>
          </div>
          <Button asChild variant="outline">
            <Link href="/">
              <ArrowLeft className="h-4 w-4" />
              返回生成页
            </Link>
          </Button>
        </header>

        <Card className="p-5 md:p-6">
          <div className="mb-5 flex items-center gap-2 text-[#006241]">
            <Store className="h-5 w-5" />
            <div>
              <h2 className="text-xl font-bold">门店基础信息</h2>
              <p className="mt-1 text-sm leading-6 text-black/58">
                配置门店定位和设备能力，后续生成配方时会作为硬边界。
              </p>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              门店名称
              <input
                value={storeConfig.storeProfile.storeName}
                onChange={(event) => {
                  setSaved(false);
                  setStoreConfig((current) => ({
                    ...current,
                    storeProfile: {
                      ...current.storeProfile,
                      storeName: event.target.value,
                    },
                  }));
                }}
                className={inputClassName}
              />
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              门店类型
              <select
                value={storeConfig.storeProfile.storeType}
                onChange={(event) => {
                  setSaved(false);
                  setStoreConfig((current) => ({
                    ...current,
                    storeProfile: {
                      ...current.storeProfile,
                      storeType: event.target.value,
                    },
                  }));
                }}
                className={inputClassName}
              >
                <option>商圈快取店</option>
                <option>社区标准店</option>
                <option>校园店</option>
                <option>景区店</option>
                <option>高端形象店</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
              品牌风格
              <input
                value={storeConfig.storeProfile.brandStyle}
                onChange={(event) => {
                  setSaved(false);
                  setStoreConfig((current) => ({
                    ...current,
                    storeProfile: {
                      ...current.storeProfile,
                      brandStyle: event.target.value,
                    },
                  }));
                }}
                className={inputClassName}
              />
            </label>
          </div>

          <div className="mt-5">
            <div className="mb-3 flex items-center gap-2 text-[#006241]">
              <SlidersHorizontal className="h-4 w-4" />
              <h3 className="font-bold">门店设备</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {equipmentOptions.map((equipment) => {
                const isSelected =
                  storeConfig.storeProfile.equipment.includes(equipment);

                return (
                  <button
                    key={equipment}
                    type="button"
                    onClick={() => toggleEquipment(equipment)}
                    className={`rounded-full border px-3 py-2 text-sm font-semibold transition ${
                      isSelected
                        ? "border-[#00754A] bg-[#d4e9e2] text-[#1E3932]"
                        : "border-border bg-white text-black/58 hover:border-[#00754A]/50"
                    }`}
                  >
                    {equipment}
                  </button>
                );
              })}
            </div>
          </div>
        </Card>

        <Card className="p-5 md:p-6">
          <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
            <div className="flex items-center gap-2 text-[#006241]">
              <Database className="h-5 w-5" />
              <div>
                <h2 className="text-xl font-bold">门店原料库</h2>
                <p className="mt-1 text-sm leading-6 text-black/58">
                  已选择 {selectedIngredients.length} 个原料，AI 只能基于这些原料组合新品。
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  setSaved(false);
                  setStoreConfig((current) => ({
                    ...current,
                    selectedIngredientIds: defaultSelectedIngredientIds,
                  }));
                }}
              >
                推荐组合
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSaved(false);
                  setStoreConfig((current) => ({
                    ...current,
                    selectedIngredientIds: [],
                  }));
                }}
              >
                清空
              </Button>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            {Object.entries(ingredientGroups).map(([category, ingredients]) => (
              <div
                key={category}
                className="rounded-xl border border-border bg-[#fbfaf7] p-4"
              >
                <div className="mb-3 flex items-center justify-between">
                  <h3 className="font-bold text-[#1E3932]">{category}</h3>
                  <span className="text-xs font-semibold text-black/50">
                    {
                      ingredients.filter((ingredient) =>
                        storeConfig.selectedIngredientIds.includes(
                          ingredient.id,
                        ),
                      ).length
                    }
                    /{ingredients.length}
                  </span>
                </div>
                <div className="grid gap-2">
                  {ingredients.map((ingredient) => {
                    const isSelected =
                      storeConfig.selectedIngredientIds.includes(ingredient.id);

                    return (
                      <button
                        key={ingredient.id}
                        type="button"
                        onClick={() => toggleIngredient(ingredient.id)}
                        className={`rounded-xl border px-3 py-3 text-left transition ${
                          isSelected
                            ? "border-[#00754A] bg-[#d4e9e2]"
                            : "border-border bg-white hover:border-[#00754A]/50"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="font-semibold text-[#1E3932]">
                            {ingredient.name}
                          </span>
                          <span
                            className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full border ${
                              isSelected
                                ? "border-[#00754A] bg-[#00754A] text-white"
                                : "border-border bg-white text-transparent"
                            }`}
                          >
                            <Check className="h-3.5 w-3.5" />
                          </span>
                        </div>
                        <div className="mt-2 flex flex-wrap gap-1.5">
                          {ingredient.flavorTags.slice(0, 3).map((tag) => (
                            <span
                              key={tag}
                              className="rounded-full bg-white/80 px-2 py-0.5 text-xs font-medium text-black/58"
                            >
                              {tag}
                            </span>
                          ))}
                          <span className="rounded-full bg-white/80 px-2 py-0.5 text-xs font-medium text-black/58">
                            {ingredient.availability}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </Card>

        <div className="sticky bottom-4 flex flex-wrap items-center justify-end gap-3 rounded-xl bg-white/92 px-4 py-3 shadow-float backdrop-blur">
          {saved ? (
            <span className="text-sm font-semibold text-[#00754A]">
              已保存门店配置
            </span>
          ) : null}
          <Button type="button" variant="outline" onClick={saveConfig}>
            <Save className="h-4 w-4" />
            保存
          </Button>
          <Button type="button" onClick={saveAndReturn}>
            <Check className="h-4 w-4" />
            保存并返回
          </Button>
        </div>
      </section>
    </main>
  );
}
