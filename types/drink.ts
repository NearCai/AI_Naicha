export type DrinkIngredient = {
  name: string;
  amount: string;
};

export type DrinkRecipe = {
  name: string;
  description: string;
  ingredients: DrinkIngredient[];
  steps: string[];
};

export type GenerateDrinkRequest = {
  prompt: string;
};

export type GenerateDrinkImageRequest = Pick<
  DrinkRecipe,
  "name" | "description" | "ingredients"
>;

export type DrinkImageResult = {
  imageUrl: string;
};

export type GenerationStatus =
  | "idle"
  | "generatingRecipe"
  | "recipeReady"
  | "generatingImage"
  | "imageReady"
  | "recipeError"
  | "imageError";
