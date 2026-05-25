import { NextResponse } from "next/server";
import type { GenerateDrinkImageRequest, DrinkImageResult } from "@/types/drink";
import { getImageModel } from "@/lib/ai-client";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
const arkImageEndpoint =
  process.env.IMAGE_API_URL ||
  "https://ark.cn-beijing.volces.com/api/v3/images/generations";

function buildImagePrompt(recipe: GenerateDrinkImageRequest): string {
  const ingredientsText = recipe.ingredients
    .map((i) => i.name)
    .join("、");

  return `一张精美的奶茶产品宣传图，饮品名称是「${recipe.name}」。
饮品描述：${recipe.description}
主要原料：${ingredientsText}

要求：
- 专业美食摄影风格，背景干净简约
- 饮品放在透明塑料杯中，杯身有品牌标签区域
- 光线柔和自然，色彩清新诱人
- 画面构图居中，突出饮品主体
- 不要出现文字、字母、logo 或菜单
- 奶茶液体、冰块、小料和顶部质感要真实
- 适合新品研发提案使用，商业产品摄影，高清细节`;
}

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

  const apiKey = process.env.IMAGE_API_KEY || process.env.VOLCENGINE_ARK_API_KEY;

  if (!apiKey) {
    // 未配置图片 AI，使用 Mock
    if (body.name.includes("图片失败")) {
      return NextResponse.json(
        { message: "图片生成服务暂时不可用，请稍后重试。" },
        { status: 500 },
      );
    }

    await sleep(1500);
    const result: DrinkImageResult = {
      imageUrl:
        "https://images.unsplash.com/photo-1558857563-b371033873b8?auto=format&fit=crop&w=1000&q=85",
    };
    return NextResponse.json(result);
  }

  try {
    const response = await fetch(arkImageEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: getImageModel(),
        prompt: buildImagePrompt(body),
        sequential_image_generation: "disabled",
        response_format: "url",
        size: process.env.IMAGE_SIZE || "2K",
        stream: false,
        watermark: process.env.IMAGE_WATERMARK !== "false",
      }),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      return NextResponse.json(
        {
          message:
            errorBody?.error?.message ||
            errorBody?.message ||
            "图片生成服务暂时不可用，请稍后重试。",
        },
        { status: response.status },
      );
    }

    const imageResponse = (await response.json()) as {
      data?: Array<{ url?: string; b64_json?: string }>;
    };
    const imageUrl = imageResponse.data?.[0]?.url;

    if (!imageUrl) {
      return NextResponse.json(
        { message: "图片生成失败，未返回图片地址。" },
        { status: 500 },
      );
    }

    const result: DrinkImageResult = { imageUrl };
    return NextResponse.json(result);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "产品图生成失败，请稍后重试。";
    return NextResponse.json({ message }, { status: 500 });
  }
}
