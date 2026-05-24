import { NextResponse } from "next/server";
import type { GenerateDrinkRequest } from "@/types/drink";

const mockRecipe = {
  name: "青提茉莉云顶奶茶",
  description:
    "一款适合夏季的清爽型奶茶，融合青提果香、茉莉茶香和轻盈奶盖。",
  ingredients: [
    { name: "茉莉绿茶汤", amount: "180g" },
    { name: "青提果汁", amount: "60g" },
    { name: "牛奶", amount: "80g" },
    { name: "冰块", amount: "120g" },
    { name: "奶盖", amount: "45g" },
  ],
  steps: [
    "杯中加入青提果汁和茉莉绿茶汤。",
    "加入牛奶和冰块后充分摇匀。",
    "倒入出品杯中。",
    "顶部加入奶盖。",
    "用青提果肉或薄荷叶装饰后出品。",
  ],
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as
    | GenerateDrinkRequest
    | null;

  if (!body?.prompt?.trim()) {
    return NextResponse.json(
      { message: "请输入饮品需求描述。" },
      { status: 400 },
    );
  }

  if (body.prompt.includes("失败")) {
    return NextResponse.json(
      { message: "Agent 暂时无法完成这条需求，请换一种描述再试。" },
      { status: 500 },
    );
  }

  await sleep(900);

  return NextResponse.json(mockRecipe);
}
