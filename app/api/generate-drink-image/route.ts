import { NextResponse } from "next/server";
import type { GenerateDrinkImageRequest } from "@/types/drink";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as
    | GenerateDrinkImageRequest
    | null;

  if (!body?.name || !body.description || !body.ingredients?.length) {
    return NextResponse.json(
      { message: "缺少生成产品图所需的饮品信息。" },
      { status: 400 },
    );
  }

  if (body.name.includes("图片失败")) {
    return NextResponse.json(
      { message: "图片生成服务暂时不可用，请稍后重试。" },
      { status: 500 },
    );
  }

  await sleep(1300);

  return NextResponse.json({
    imageUrl:
      "https://images.unsplash.com/photo-1558857563-b371033873b8?auto=format&fit=crop&w=1000&q=85",
  });
}
