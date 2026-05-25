"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Check,
  Database,
  Plus,
  Save,
  SlidersHorizontal,
  Store,
  Trash2,
  CheckCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  defaultSelectedIngredientIds,
  equipmentOptions,
  getAllIngredients,
  getDefaultStoreConfig,
  ingredientLibrary,
  normalizeStoreConfig,
  storeConfigStorageKey,
} from "@/lib/store-config";
import type { StoreConfig } from "@/lib/store-config";
import type { StoreIngredient } from "@/types/drink";

const inputClassName =
  "h-11 w-full rounded-xl border border-input bg-white px-3 text-sm font-medium text-foreground shadow-sm outline-none transition focus:ring-2 focus:ring-ring";

const smallInputClassName =
  "h-8 w-full rounded-lg border border-input bg-white px-2 text-sm font-medium text-foreground shadow-sm outline-none transition focus:ring-2 focus:ring-ring";

export default function StoreConfigPage() {
  const router = useRouter();
  const [storeConfig, setStoreConfig] = useState<StoreConfig>(
    getDefaultStoreConfig,
  );
  const [saved, setSaved] = useState(false);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newIngredient, setNewIngredient] = useState({
    name: "",
    category: "茶底" as StoreIngredient["category"],
    quantity: "1000g",
    costPerUnit: "",
    flavorTags: "",
    availability: "充足" as StoreIngredient["availability"],
  });

  const allIngredients = useMemo(
    () => getAllIngredients(storeConfig),
    [storeConfig],
  );

  const selectedIngredients = useMemo(
    () =>
      allIngredients.filter((ingredient) =>
        storeConfig.selectedIngredientIds.includes(ingredient.id),
      ),
    [allIngredients, storeConfig.selectedIngredientIds],
  );

  const ingredientGroups = useMemo(
    () =>
      allIngredients.reduce(
        (groups, ingredient) => {
          groups[ingredient.category] = [
            ...(groups[ingredient.category] ?? []),
            ingredient,
          ];
          return groups;
        },
        {} as Record<StoreIngredient["category"], StoreIngredient[]>,
      ),
    [allIngredients],
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

  function updateQuantity(id: string, quantity: string) {
    setSaved(false);
    setStoreConfig((current) => ({
      ...current,
      ingredientQuantities: {
        ...current.ingredientQuantities,
        [id]: quantity,
      },
    }));
  }

  function selectAll() {
    setSaved(false);
    setStoreConfig((current) => ({
      ...current,
      selectedIngredientIds: allIngredients.map((i) => i.id),
    }));
  }

  function addCustomIngredient() {
    const trimmedName = newIngredient.name.trim();
    if (!trimmedName) {
      return;
    }

    const id = `custom-${Date.now()}`;
    const ingredient: StoreIngredient = {
      id,
      name: trimmedName,
      category: newIngredient.category,
      quantity: newIngredient.quantity.trim() || "1000g",
      costPerUnit: newIngredient.costPerUnit.trim() || "0元",
      flavorTags: newIngredient.flavorTags
        .split(/[,，]/)
        .map((s) => s.trim())
        .filter(Boolean),
      allergens: [],
      availability: newIngredient.availability,
      equipment: [],
    };

    setSaved(false);
    setStoreConfig((current) => ({
      ...current,
      customIngredients: [...current.customIngredients, ingredient],
      selectedIngredientIds: [...current.selectedIngredientIds, id],
      ingredientQuantities: {
        ...current.ingredientQuantities,
        [id]: ingredient.quantity,
      },
    }));

    setNewIngredient({
      name: "",
      category: "茶底",
      quantity: "1000g",
      costPerUnit: "",
      flavorTags: "",
      availability: "充足",
    });
    setShowAddForm(false);
  }

  function removeCustomIngredient(id: string) {
    setSaved(false);
    setStoreConfig((current) => {
      const nextQuantities = { ...current.ingredientQuantities };
      delete nextQuantities[id];

      return {
        ...current,
        customIngredients: current.customIngredients.filter(
          (i) => i.id !== id,
        ),
        selectedIngredientIds: current.selectedIngredientIds.filter(
          (sid) => sid !== id,
        ),
        ingredientQuantities: nextQuantities,
      };
    });
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
                  已选择 {selectedIngredients.length} 个原料，AI
                  只能基于这些原料组合新品。
                </p>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
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
                variant="outline"
                size="sm"
                onClick={selectAll}
              >
                <CheckCheck className="h-4 w-4" />
                全选
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
                      storeConfig.selectedIngredientIds.includes(
                        ingredient.id,
                      );
                    const isCustom = ingredient.id.startsWith("custom-");

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
                          <div className="flex items-center gap-2">
                            {isCustom ? (
                              <span
                                className="flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-full text-red-500 hover:bg-red-50"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  removeCustomIngredient(ingredient.id);
                                }}
                                title="删除自定义原料"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </span>
                            ) : null}
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
                        <div
                          className="mt-2"
                          onClick={(event) => event.stopPropagation()}
                          onPointerDown={(event) => event.stopPropagation()}
                        >
                          <label className="flex items-center gap-2 text-xs text-black/50">
                            <span className="shrink-0">数量</span>
                            <input
                              value={
                                storeConfig.ingredientQuantities[
                                  ingredient.id
                                ] ??
                                ingredientLibrary.find(
                                  (i) => i.id === ingredient.id,
                                )?.quantity ??
                                ingredient.quantity
                              }
                              onChange={(event) =>
                                updateQuantity(
                                  ingredient.id,
                                  event.target.value,
                                )
                              }
                              className={`${smallInputClassName} w-24`}
                              placeholder="如 500g"
                            />
                          </label>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-6">
            {!showAddForm ? (
              <Button
                type="button"
                variant="outline"
                onClick={() => setShowAddForm(true)}
              >
                <Plus className="h-4 w-4" />
                添加自定义原料
              </Button>
            ) : (
              <div className="rounded-xl border border-border bg-[#fbfaf7] p-4 md:p-5">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="font-bold text-[#1E3932]">添加自定义原料</h3>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAddForm(false)}
                  >
                    取消
                  </Button>
                </div>
                <div className="grid gap-4 md:grid-cols-3">
                  <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
                    原料名称
                    <input
                      value={newIngredient.name}
                      onChange={(event) =>
                        setNewIngredient((current) => ({
                          ...current,
                          name: event.target.value,
                        }))
                      }
                      placeholder="例如：白桃乌龙茶"
                      className={inputClassName}
                    />
                  </label>
                  <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
                    分类
                    <select
                      value={newIngredient.category}
                      onChange={(event) =>
                        setNewIngredient((current) => ({
                          ...current,
                          category: event.target
                            .value as StoreIngredient["category"],
                        }))
                      }
                      className={inputClassName}
                    >
                      <option>茶底</option>
                      <option>奶基底</option>
                      <option>水果</option>
                      <option>小料</option>
                      <option>风味糖浆</option>
                      <option>辅料</option>
                    </select>
                  </label>
                  <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
                    数量
                    <input
                      value={newIngredient.quantity}
                      onChange={(event) =>
                        setNewIngredient((current) => ({
                          ...current,
                          quantity: event.target.value,
                        }))
                      }
                      placeholder="如 1000g / 2000ml"
                      className={inputClassName}
                    />
                  </label>
                  <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
                    单位成本
                    <input
                      value={newIngredient.costPerUnit}
                      onChange={(event) =>
                        setNewIngredient((current) => ({
                          ...current,
                          costPerUnit: event.target.value,
                        }))
                      }
                      placeholder="如 0.02元/ml"
                      className={inputClassName}
                    />
                  </label>
                  <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
                    风味标签
                    <input
                      value={newIngredient.flavorTags}
                      onChange={(event) =>
                        setNewIngredient((current) => ({
                          ...current,
                          flavorTags: event.target.value,
                        }))
                      }
                      placeholder="用逗号分隔，如：花香，清爽"
                      className={inputClassName}
                    />
                  </label>
                  <label className="grid gap-2 text-sm font-semibold text-[#1E3932]">
                    供应状态
                    <select
                      value={newIngredient.availability}
                      onChange={(event) =>
                        setNewIngredient((current) => ({
                          ...current,
                          availability: event.target
                            .value as StoreIngredient["availability"],
                        }))
                      }
                      className={inputClassName}
                    >
                      <option>充足</option>
                      <option>偏低</option>
                      <option>季节限定</option>
                    </select>
                  </label>
                </div>
                <div className="mt-4 flex justify-end">
                  <Button type="button" onClick={addCustomIngredient}>
                    <Plus className="h-4 w-4" />
                    确认添加
                  </Button>
                </div>
              </div>
            )}
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
