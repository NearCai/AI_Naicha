import { NextResponse } from "next/server";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import type {
  AddHistoryFeedbackRequest,
  DrinkFeedback,
} from "@/types/drink";

type HistoryRecord = {
  id: string;
  feedbacks?: DrinkFeedback[];
  userRatings?: number[];
  averageRating?: number;
  sales?: {
    weeklyCups: number;
    monthlyCups: number;
    conversionRate: number;
    repeatRate: number;
  };
};

type HistoryJson = {
  records: HistoryRecord[];
};

function normalizeScore(value: unknown) {
  const score = Number(value);

  if (!Number.isFinite(score)) {
    return null;
  }

  return Math.max(0, Math.min(10, score));
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

function normalizeExistingFeedbacks(record: HistoryRecord) {
  if (Array.isArray(record.feedbacks)) {
    return record.feedbacks;
  }

  return Array.isArray(record.userRatings)
    ? record.userRatings.map((score, index) => ({
        score,
        comment: `历史评分 ${index + 1}`,
        createdAt: "",
      }))
    : [];
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as
    | AddHistoryFeedbackRequest
    | null;
  const score = normalizeScore(body?.feedback?.score);
  const comment = String(body?.feedback?.comment ?? "").trim();

  if (!body?.id) {
    return NextResponse.json({ message: "缺少历史记录 ID。" }, { status: 400 });
  }

  if (score === null) {
    return NextResponse.json(
      { message: "请填写 0-10 分的有效分数。" },
      { status: 400 },
    );
  }

  const historyDir = path.join(process.cwd(), "data");
  const historyPath = path.join(historyDir, "recipe-history.json");
  await mkdir(historyDir, { recursive: true });

  let history: HistoryJson;
  try {
    history = JSON.parse(await readFile(historyPath, "utf-8")) as HistoryJson;
  } catch {
    return NextResponse.json({ message: "暂无历史数据。" }, { status: 404 });
  }

  const record = history.records.find((item) => item.id === body.id);

  if (!record) {
    return NextResponse.json(
      { message: "未找到对应历史配方。" },
      { status: 404 },
    );
  }

  const feedbacks = [
    ...normalizeExistingFeedbacks(record),
    {
      score,
      comment,
      createdAt: new Date().toISOString(),
    },
  ];
  const scores = feedbacks.map((feedback) => feedback.score);
  const averageRating = Number(average(scores).toFixed(2));

  record.feedbacks = feedbacks;
  record.userRatings = scores;
  record.averageRating = averageRating;
  record.sales = estimateSales(averageRating, scores);

  await writeFile(historyPath, `${JSON.stringify(history, null, 2)}\n`, "utf-8");

  return NextResponse.json({
    id: body.id,
    averageRating,
    feedbackCount: feedbacks.length,
    sales: record.sales,
  });
}
