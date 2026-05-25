export type DrinkIngredient = {
  name: string;
  amount: string;
};

export type StoreIngredient = {
  id: string;
  name: string;
  category: "茶底" | "奶基底" | "水果" | "小料" | "风味糖浆" | "辅料";
  costPerUnit: string;
  quantity: string;
  flavorTags: string[];
  allergens: string[];
  availability: "充足" | "偏低" | "季节限定";
  equipment: string[];
};

export type DrinkRecipe = {
  name: string;
  description: string;
  ingredients: DrinkIngredient[];
  steps: string[];
};

export type RecipeFilterReport = {
  keptCount: number;
  rejectedCount: number;
  rejected: Array<{
    name: string;
    reasons: string[];
  }>;
};

export type AuditResult = {
  auditorName: string;
  selectedRecipeName: string;
  summary: string;
  reasons: string[];
  marketSignals: string[];
};

export type DrinkDevelopmentResult = {
  engineerName: string;
  skillReferences: string[];
  recipes: DrinkRecipe[];
};

export type DrinkAuditStageResult = {
  recipes: DrinkRecipe[];
  filterReport: RecipeFilterReport;
  audit: AuditResult;
};

export type DrinkGenerationResult = DrinkDevelopmentResult & DrinkAuditStageResult;

export type StoreProfile = {
  storeName: string;
  storeType: string;
  brandStyle: string;
  equipment: string[];
};

export type GenerationConstraints = {
  season: string;
  targetAudience: string;
  priceBand: string;
  maxIngredientCost: string;
  maxMakeTime: string;
  sweetness: string;
  temperature: string;
};

export type GenerateDrinkRequest = {
  prompt: string;
  generationCount?: number;
  storeProfile?: StoreProfile;
  constraints?: GenerationConstraints;
  availableIngredients?: StoreIngredient[];
};

export type AuditDrinkRequest = {
  recipes: DrinkRecipe[];
  constraints?: GenerationConstraints;
  availableIngredients?: StoreIngredient[];
};

export type GenerateDrinkImageRequest = Pick<
  DrinkRecipe,
  "name" | "description" | "ingredients"
>;

export type DrinkImageResult = {
  imageUrl: string;
};

export type DrinkFeedback = {
  score: number;
  comment: string;
  createdAt?: string;
};

export type RateDrinkRequest = {
  recipe: DrinkRecipe;
  feedbacks: DrinkFeedback[];
  constraints?: GenerationConstraints;
};

export type RateDrinkResult = {
  id: string;
  averageRating: number;
  library: "recipe_skill_library" | "bad_recipe_skill_library";
  savedTo: string[];
};

export type AddHistoryFeedbackRequest = {
  id: string;
  feedback: DrinkFeedback;
};

export type AddHistoryFeedbackResult = {
  id: string;
  averageRating: number;
  feedbackCount: number;
  sales: {
    weeklyCups: number;
    monthlyCups: number;
    conversionRate: number;
    repeatRate: number;
  };
};

export type GenerationStatus =
  | "idle"
  | "generatingDevelopment"
  | "developmentReady"
  | "auditing"
  | "auditReady"
  | "recipeError"
  | "generatingImage"
  | "imageReady"
  | "imageError";
