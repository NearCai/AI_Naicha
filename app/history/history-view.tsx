"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  BarChart3,
  CalendarClock,
  CircleDollarSign,
  Search,
  Star,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { addHistoryFeedback } from "@/lib/api";
import type {
  DrinkFeedback,
  DrinkRecipe,
  GenerationConstraints,
} from "@/types/drink";

type SalesMetrics = {
  weeklyCups: number;
  monthlyCups: number;
  conversionRate: number;
  repeatRate: number;
};

export type HistoryRecord = {
  id: string;
  createdAt: string;
  recipe: DrinkRecipe;
  constraints: GenerationConstraints | null;
  feedbacks?: DrinkFeedback[];
  userRatings: number[];
  averageRating: number;
  library: "recipe_skill_library" | "bad_recipe_skill_library";
  sales: SalesMetrics;
  skillFiles: string[];
};

type HistoryViewProps = {
  records: HistoryRecord[];
};

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function feedbacksForRecord(record: HistoryRecord) {
  if (Array.isArray(record.feedbacks) && record.feedbacks.length) {
    return record.feedbacks;
  }

  return record.userRatings.map((score, index) => ({
    score,
    comment: `历史评分 ${index + 1}`,
    createdAt: "",
  }));
}

export function HistoryView({ records }: HistoryViewProps) {
  const [historyRecords, setHistoryRecords] = useState(records);
  const [query, setQuery] = useState("");
  const [libraryFilter, setLibraryFilter] = useState("全部");
  const [sortBy, setSortBy] = useState("最新");
  const [drafts, setDrafts] = useState<Record<string, DrinkFeedback>>({});
  const [savingId, setSavingId] = useState("");
  const [feedbackMessages, setFeedbackMessages] = useState<Record<string, string>>(
    {},
  );

  const filteredRecords = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return historyRecords
      .filter((record) => {
        const matchesLibrary =
          libraryFilter === "全部" ||
          (libraryFilter === "好配方" &&
            record.library === "recipe_skill_library") ||
          (libraryFilter === "差配方" &&
            record.library === "bad_recipe_skill_library");
        const haystack = [
          record.recipe.name,
          record.recipe.description,
          record.id,
          record.constraints?.season ?? "",
          record.constraints?.targetAudience ?? "",
          ...record.recipe.ingredients.map((ingredient) => ingredient.name),
        ]
          .join(" ")
          .toLowerCase();

        return matchesLibrary && (!normalizedQuery || haystack.includes(normalizedQuery));
      })
      .sort((a, b) => {
        if (sortBy === "评分最高") {
          return b.averageRating - a.averageRating;
        }
        if (sortBy === "销量最高") {
          return b.sales.monthlyCups - a.sales.monthlyCups;
        }
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });
  }, [historyRecords, libraryFilter, query, sortBy]);

  const totalMonthlyCups = historyRecords.reduce(
    (sum, record) => sum + record.sales.monthlyCups,
    0,
  );
  const averageScore = average(historyRecords.map((record) => record.averageRating));
  const highScoreCount = historyRecords.filter(
    (record) => record.library === "recipe_skill_library",
  ).length;

  function draftForRecord(recordId: string) {
    return drafts[recordId] ?? { score: 8, comment: "" };
  }

  function updateDraft(recordId: string, nextDraft: DrinkFeedback) {
    setDrafts((current) => ({
      ...current,
      [recordId]: nextDraft,
    }));
    setFeedbackMessages((current) => ({ ...current, [recordId]: "" }));
  }

  async function submitFeedback(record: HistoryRecord) {
    const draft = draftForRecord(record.id);
    setSavingId(record.id);
    setFeedbackMessages((current) => ({ ...current, [record.id]: "" }));

    try {
      const result = await addHistoryFeedback({
        id: record.id,
        feedback: draft,
      });
      const nextFeedback = {
        score: Math.max(0, Math.min(10, Number(draft.score) || 0)),
        comment: draft.comment.trim(),
        createdAt: new Date().toISOString(),
      };

      setHistoryRecords((current) =>
        current.map((item) => {
          if (item.id !== record.id) {
            return item;
          }

          const feedbacks = [...feedbacksForRecord(item), nextFeedback];
          const scores = feedbacks.map((feedback) => feedback.score);

          return {
            ...item,
            feedbacks,
            userRatings: scores,
            averageRating: result.averageRating,
            sales: result.sales,
          };
        }),
      );
      setDrafts((current) => ({
        ...current,
        [record.id]: { score: 8, comment: "" },
      }));
      setFeedbackMessages((current) => ({
        ...current,
        [record.id]: `已新增反馈，当前共 ${result.feedbackCount} 条，平均分 ${result.averageRating.toFixed(1)}。`,
      }));
    } catch (error) {
      setFeedbackMessages((current) => ({
        ...current,
        [record.id]:
          error instanceof Error ? error.message : "反馈保存失败，请稍后重试。",
      }));
    } finally {
      setSavingId("");
    }
  }

  return (
    <main className="min-h-screen bg-[#f2f0eb]">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-4 py-6 md:px-8 lg:px-10">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-xl bg-white px-5 py-4 shadow-soft">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#00754A]">
              Recipe History
            </p>
            <h1 className="mt-1 text-2xl font-black text-[#006241] md:text-3xl">
              历史配方回溯
            </h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline">
              <Link href="/skill-library">
                <BarChart3 className="h-4 w-4" />
                Skill 库
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

        <div className="grid gap-3 md:grid-cols-4">
          <Card className="px-4 py-4">
            <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.12em] text-black/45">
              <CalendarClock className="h-4 w-4" />
              Records
            </p>
            <p className="mt-2 text-3xl font-black text-[#1E3932]">
              {historyRecords.length}
            </p>
          </Card>
          <Card className="px-4 py-4">
            <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.12em] text-black/45">
              <Star className="h-4 w-4" />
              Avg Score
            </p>
            <p className="mt-2 text-3xl font-black text-[#00754A]">
              {averageScore.toFixed(1)}
            </p>
          </Card>
          <Card className="px-4 py-4">
            <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.12em] text-black/45">
              <TrendingUp className="h-4 w-4" />
              Good
            </p>
            <p className="mt-2 text-3xl font-black text-[#00754A]">
              {highScoreCount}
            </p>
          </Card>
          <Card className="px-4 py-4">
            <p className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.12em] text-black/45">
              <CircleDollarSign className="h-4 w-4" />
              Monthly Cups
            </p>
            <p className="mt-2 text-3xl font-black text-[#1E3932]">
              {totalMonthlyCups}
            </p>
          </Card>
        </div>

        <Card className="p-4 md:p-5">
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_160px_160px]">
            <label className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-black/36" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="h-11 w-full rounded-xl border border-input bg-white pl-9 pr-3 text-sm font-medium outline-none transition focus:ring-2 focus:ring-ring"
                placeholder="搜索配方、原料、客群、ID"
              />
            </label>
            <select
              value={libraryFilter}
              onChange={(event) => setLibraryFilter(event.target.value)}
              className="h-11 rounded-xl border border-input bg-white px-3 text-sm font-bold text-[#1E3932] outline-none focus:ring-2 focus:ring-ring"
            >
              <option>全部</option>
              <option>好配方</option>
              <option>差配方</option>
            </select>
            <select
              value={sortBy}
              onChange={(event) => setSortBy(event.target.value)}
              className="h-11 rounded-xl border border-input bg-white px-3 text-sm font-bold text-[#1E3932] outline-none focus:ring-2 focus:ring-ring"
            >
              <option>最新</option>
              <option>评分最高</option>
              <option>销量最高</option>
            </select>
          </div>
        </Card>

        <div className="grid gap-4">
          {filteredRecords.map((record) => {
            const isGood = record.library === "recipe_skill_library";
            const feedbacks = feedbacksForRecord(record);
            const draft = draftForRecord(record.id);
            const isSaving = savingId === record.id;

            return (
              <Card key={record.id} className="overflow-hidden">
                <div
                  className={`px-5 py-4 text-white ${
                    isGood ? "bg-[#00754A]" : "bg-[#c2410c]"
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <p className="text-xs font-bold uppercase tracking-[0.12em] text-white/70">
                        {formatDate(record.createdAt)} · {record.id}
                      </p>
                      <h2 className="mt-1 text-2xl font-black">
                        {record.recipe.name}
                      </h2>
                    </div>
                    <div className="rounded-full bg-white px-3 py-1 text-sm font-black text-[#1E3932]">
                      {isGood ? "好配方库" : "差配方库"}
                    </div>
                  </div>
                  <p className="mt-3 max-w-4xl text-sm leading-7 text-white/80">
                    {record.recipe.description}
                  </p>
                </div>

                <div className="grid gap-4 p-5 lg:grid-cols-[minmax(0,1fr)_320px]">
                  <div>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {record.recipe.ingredients.map((ingredient, index) => (
                        <div
                          key={`${record.id}-${ingredient.name}-${index}`}
                          className="flex items-center justify-between gap-3 rounded-lg border border-border bg-[#fbfaf7] px-3 py-2"
                        >
                          <span className="font-semibold text-black/70">
                            {ingredient.name}
                          </span>
                          <span className="font-black text-[#00754A]">
                            {ingredient.amount}
                          </span>
                        </div>
                      ))}
                    </div>

                    <div className="mt-4 grid gap-2">
                      {feedbacks.map((feedback, index) => (
                        <div
                          key={`${record.id}-rating-${index}`}
                          className="rounded-lg bg-[#edebe9] px-3 py-2"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <p className="text-xs font-bold text-black/45">
                              反馈 {index + 1}
                            </p>
                            <p className="text-lg font-black text-[#1E3932]">
                              {feedback.score}
                            </p>
                          </div>
                          <p className="mt-1 text-sm leading-6 text-black/65">
                            {feedback.comment || "未填写文字评价"}
                          </p>
                        </div>
                      ))}
                    </div>

                    <div className="mt-4 rounded-xl border border-border bg-white p-3">
                      <h3 className="font-black text-[#006241]">新增反馈</h3>
                      <div className="mt-3 grid gap-3 sm:grid-cols-[96px_minmax(0,1fr)]">
                        <label className="grid gap-1 text-xs font-bold text-[#1E3932]">
                          分数
                          <input
                            type="number"
                            min={0}
                            max={10}
                            step={1}
                            value={draft.score}
                            onChange={(event) =>
                              updateDraft(record.id, {
                                ...draft,
                                score: Math.max(
                                  0,
                                  Math.min(10, Number(event.target.value) || 0),
                                ),
                              })
                            }
                            disabled={isSaving}
                            className="h-10 rounded-lg border border-input bg-white px-2 text-center text-sm font-black outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
                          />
                        </label>
                        <label className="grid gap-1 text-xs font-bold text-[#1E3932]">
                          评价
                          <input
                            value={draft.comment}
                            onChange={(event) =>
                              updateDraft(record.id, {
                                ...draft,
                                comment: event.target.value,
                              })
                            }
                            disabled={isSaving}
                            placeholder="补充新的试饮反馈"
                            className="h-10 rounded-lg border border-input bg-white px-3 text-sm font-medium outline-none focus:ring-2 focus:ring-ring disabled:opacity-50"
                          />
                        </label>
                      </div>
                      <Button
                        type="button"
                        className="mt-3 w-full"
                        onClick={() => submitFeedback(record)}
                        disabled={isSaving}
                      >
                        {isSaving ? "保存中" : "新增反馈"}
                      </Button>
                      {feedbackMessages[record.id] ? (
                        <p className="mt-3 rounded-xl bg-[#d4e9e2] px-3 py-2 text-sm font-semibold leading-6 text-[#1E3932]">
                          {feedbackMessages[record.id]}
                        </p>
                      ) : null}
                    </div>
                  </div>

                  <div className="grid gap-3">
                    <div className="rounded-xl border border-border bg-[#fbfaf7] px-4 py-3">
                      <p className="text-xs font-bold uppercase tracking-[0.12em] text-black/45">
                        Average Rating
                      </p>
                      <p className="mt-1 text-4xl font-black text-[#00754A]">
                        {record.averageRating.toFixed(1)}
                      </p>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded-xl border border-border bg-white px-3 py-3">
                        <p className="text-xs font-bold text-black/45">周销量</p>
                        <p className="mt-1 text-2xl font-black text-[#1E3932]">
                          {record.sales.weeklyCups}
                        </p>
                      </div>
                      <div className="rounded-xl border border-border bg-white px-3 py-3">
                        <p className="text-xs font-bold text-black/45">月销量</p>
                        <p className="mt-1 text-2xl font-black text-[#1E3932]">
                          {record.sales.monthlyCups}
                        </p>
                      </div>
                      <div className="rounded-xl border border-border bg-white px-3 py-3">
                        <p className="text-xs font-bold text-black/45">转化率</p>
                        <p className="mt-1 text-2xl font-black text-[#1E3932]">
                          {percent(record.sales.conversionRate)}
                        </p>
                      </div>
                      <div className="rounded-xl border border-border bg-white px-3 py-3">
                        <p className="text-xs font-bold text-black/45">复购率</p>
                        <p className="mt-1 text-2xl font-black text-[#1E3932]">
                          {percent(record.sales.repeatRate)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}

          {!filteredRecords.length ? (
            <Card className="grid min-h-[360px] place-items-center p-8 text-center">
              <div>
                <h2 className="text-2xl font-black text-[#006241]">
                  还没有历史配方
                </h2>
                <p className="mt-3 max-w-md text-sm leading-7 text-black/58">
                  生成最终配方后，在反馈收集模块提交分数和评价，系统会自动把配方、反馈和销量指标写入历史回溯。
                </p>
              </div>
            </Card>
          ) : null}
        </div>
      </section>
    </main>
  );
}
