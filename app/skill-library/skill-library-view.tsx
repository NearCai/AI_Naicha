"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  BookOpen,
  CalendarClock,
  CircleDollarSign,
  Clock3,
  Flame,
  Search,
  Snowflake,
  Tag,
  ThermometerSun,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export type SkillIngredient = {
  name: string;
  amount: string;
};

export type SkillRecipe = {
  section: string;
  name: string;
  id: string;
  style: string;
  volume: string;
  sweetness: string;
  estimatedCostYuan: number | null;
  ingredients: SkillIngredient[];
  problem?: string;
};

export type SkillCategory = {
  key: string;
  label: string;
  sourceMarkdown: string;
  recipeCount: number;
  recipes: SkillRecipe[];
};

export type SkillLibrary = {
  key: "good" | "bad";
  title: string;
  subtitle: string;
  tone: "good" | "bad";
  categories: SkillCategory[];
};

type SkillLibraryViewProps = {
  libraries: SkillLibrary[];
};

const categoryIcons: Record<string, typeof BookOpen> = {
  season: ThermometerSun,
  price_band: Tag,
  cost_cap: CircleDollarSign,
  make_time: Clock3,
  target_audience: Users,
  sweetness: Flame,
  temperature: Snowflake,
};

const libraryShellClass = {
  good: "border-[#b7d7cb] bg-[#f6fbf8]",
  bad: "border-[#f0c6bd] bg-[#fff7f4]",
};

const libraryAccentClass = {
  good: "bg-[#00754A] text-white",
  bad: "bg-[#c2410c] text-white",
};

function formatCost(cost: number | null) {
  return typeof cost === "number" ? `¥${cost.toFixed(2)}` : "未估算";
}

function categorySections(category: SkillCategory) {
  return Array.from(new Set(category.recipes.map((recipe) => recipe.section)));
}

function LibraryPanel({ library }: { library: SkillLibrary }) {
  const [categoryKey, setCategoryKey] = useState(
    library.categories[0]?.key ?? "",
  );
  const [section, setSection] = useState("全部");
  const [query, setQuery] = useState("");

  const selectedCategory =
    library.categories.find((category) => category.key === categoryKey) ??
    library.categories[0];
  const sections = selectedCategory ? categorySections(selectedCategory) : [];

  const visibleRecipes = useMemo(() => {
    if (!selectedCategory) {
      return [];
    }

    const trimmedQuery = query.trim().toLowerCase();

    return selectedCategory.recipes.filter((recipe) => {
      const matchesSection = section === "全部" || recipe.section === section;
      const haystack = [
        recipe.name,
        recipe.id,
        recipe.style,
        recipe.section,
        recipe.problem ?? "",
        ...recipe.ingredients.map((ingredient) => ingredient.name),
      ]
        .join(" ")
        .toLowerCase();

      return matchesSection && (!trimmedQuery || haystack.includes(trimmedQuery));
    });
  }, [query, section, selectedCategory]);

  function selectCategory(nextCategoryKey: string) {
    setCategoryKey(nextCategoryKey);
    setSection("全部");
    setQuery("");
  }

  return (
    <section
      className={`min-w-0 rounded-2xl border p-4 md:p-5 ${libraryShellClass[library.tone]}`}
    >
      <div className={`rounded-xl px-4 py-4 ${libraryAccentClass[library.tone]}`}>
        <p className="text-xs font-bold uppercase tracking-[0.12em] text-white/70">
          {library.key === "good" ? "正向技能" : "反向技能"}
        </p>
        <h2 className="mt-1 text-2xl font-black">{library.title}</h2>
        <p className="mt-2 text-sm leading-6 text-white/78">{library.subtitle}</p>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl bg-white px-4 py-3 shadow-soft">
          <p className="text-xs font-bold uppercase tracking-[0.12em] text-black/45">
            分类数
          </p>
          <p className="mt-1 text-2xl font-black text-[#1E3932]">
            {library.categories.length}
          </p>
        </div>
        <div className="rounded-xl bg-white px-4 py-3 shadow-soft">
          <p className="text-xs font-bold uppercase tracking-[0.12em] text-black/45">
            当前分类
          </p>
          <p className="mt-1 text-2xl font-black text-[#1E3932]">
            {selectedCategory?.recipeCount ?? 0}
          </p>
        </div>
        <div className="rounded-xl bg-white px-4 py-3 shadow-soft">
          <p className="text-xs font-bold uppercase tracking-[0.12em] text-black/45">
            当前展示
          </p>
          <p className="mt-1 text-2xl font-black text-[#1E3932]">
            {visibleRecipes.length}
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {library.categories.map((category) => {
          const Icon = categoryIcons[category.key] ?? BookOpen;
          const selected = category.key === categoryKey;

          return (
            <button
              key={category.key}
              type="button"
              onClick={() => selectCategory(category.key)}
              className={`flex min-h-16 items-center gap-2 rounded-xl border px-3 py-2 text-left transition ${
                selected
                  ? "border-[#00754A] bg-white text-[#006241] shadow-soft"
                  : "border-border bg-white/70 text-black/62 hover:border-[#00754A]/50"
              }`}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="min-w-0">
                <span className="block text-sm font-black">
                  {category.label}
                </span>
                <span className="block text-xs font-semibold text-black/48">
                  {category.recipeCount} 条
                </span>
              </span>
            </button>
          );
        })}
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_180px]">
        <label className="relative block">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-black/36" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="h-11 w-full rounded-xl border border-input bg-white pl-9 pr-3 text-sm font-medium outline-none transition focus:ring-2 focus:ring-ring"
            placeholder="搜索配方、原料、问题"
          />
        </label>
        <select
          value={section}
          onChange={(event) => setSection(event.target.value)}
          className="h-11 w-full rounded-xl border border-input bg-white px-3 text-sm font-bold text-[#1E3932] outline-none transition focus:ring-2 focus:ring-ring"
        >
          <option>全部</option>
          {sections.map((currentSection) => (
            <option key={currentSection}>{currentSection}</option>
          ))}
        </select>
      </div>

      <div className="mt-4 grid max-h-[920px] gap-3 overflow-y-auto pr-1">
        {visibleRecipes.map((recipe) => (
          <Card
            key={`${selectedCategory?.key}-${recipe.id}-${recipe.section}`}
            className="overflow-hidden border border-white/80"
          >
            <div className="border-b border-border bg-white px-4 py-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.1em] text-[#00754A]">
                    {recipe.section} · {recipe.id}
                  </p>
                  <h3 className="mt-1 text-lg font-black text-[#1E3932]">
                    {recipe.name}
                  </h3>
                </div>
                <div className="flex flex-wrap gap-1.5 text-xs font-bold">
                  <span className="rounded-full bg-[#d4e9e2] px-2.5 py-1 text-[#1E3932]">
                    {recipe.style}
                  </span>
                  <span className="rounded-full bg-[#edebe9] px-2.5 py-1 text-black/62">
                    {recipe.volume}
                  </span>
                  <span className="rounded-full bg-[#edebe9] px-2.5 py-1 text-black/62">
                    {recipe.sweetness}
                  </span>
                  <span className="rounded-full bg-[#edebe9] px-2.5 py-1 text-black/62">
                    {formatCost(recipe.estimatedCostYuan)}
                  </span>
                </div>
              </div>
              {recipe.problem ? (
                <p className="mt-3 rounded-lg bg-[#fff1eb] px-3 py-2 text-sm font-medium leading-6 text-[#9a3412]">
                  {recipe.problem}
                </p>
              ) : null}
            </div>

            <div className="grid gap-2 bg-[#fbfaf7] p-4 sm:grid-cols-2">
              {recipe.ingredients.map((ingredient, index) => (
                <div
                  key={`${recipe.id}-${ingredient.name}-${index}`}
                  className="flex items-center justify-between gap-3 rounded-lg border border-border bg-white px-3 py-2"
                >
                  <span className="min-w-0 truncate text-sm font-semibold text-black/70">
                    {ingredient.name}
                  </span>
                  <span className="shrink-0 text-sm font-black text-[#00754A]">
                    {ingredient.amount}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        ))}

        {!visibleRecipes.length ? (
          <div className="grid min-h-40 place-items-center rounded-xl border border-dashed border-border bg-white/70 px-6 text-center text-sm font-semibold text-black/50">
            没有匹配的配方条目
          </div>
        ) : null}
      </div>
    </section>
  );
}

export function SkillLibraryView({ libraries }: SkillLibraryViewProps) {
  return (
    <main className="min-h-screen bg-[#f2f0eb]">
      <section className="mx-auto flex w-full max-w-[1720px] flex-col gap-5 px-4 py-6 md:px-8 lg:px-10">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-xl bg-white px-5 py-4 shadow-soft">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
              配方技能库
            </p>
            <h1 className="mt-1 text-2xl font-black text-[#006241] md:text-3xl">
              奶茶 Skill 库可视化
            </h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline">
              <Link href="/history">
                <CalendarClock className="h-4 w-4" />
                历史回溯
              </Link>
            </Button>
            <Button asChild variant="outline">
              <Link href="/">
                <ArrowLeft className="h-4 w-4" />
                返回生成页
              </Link>
            </Button>
          </div>
        </header>

        <div className="grid gap-5 xl:grid-cols-2">
          {libraries.map((library) => (
            <LibraryPanel key={library.key} library={library} />
          ))}
        </div>
      </section>
    </main>
  );
}
