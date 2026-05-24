import type {
  DrinkImageResult,
  DrinkRecipe,
  GenerateDrinkImageRequest,
  GenerateDrinkRequest,
} from "@/types/drink";

export async function generateDrinkRecipe(
  payload: GenerateDrinkRequest,
): Promise<DrinkRecipe> {
  const response = await fetch("/api/generate-drink", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.message ?? "配方生成失败，请稍后重试。");
  }

  return response.json();
}

export async function generateDrinkImage(
  payload: GenerateDrinkImageRequest,
): Promise<DrinkImageResult> {
  const response = await fetch("/api/generate-drink-image", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);
    throw new Error(errorBody?.message ?? "产品图生成失败，请稍后重试。");
  }

  return response.json();
}
