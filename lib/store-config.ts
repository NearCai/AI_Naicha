import type { StoreIngredient, StoreProfile } from "@/types/drink";

export const storeConfigStorageKey = "ai-drink-lab-store-config";

export const ingredientLibrary: StoreIngredient[] = [
  {
    id: "jasmine_green_tea",
    name: "茉莉绿茶汤",
    category: "茶底",
    costPerUnit: "0.018元/ml",
    flavorTags: ["花香", "清爽", "茶感"],
    allergens: [],
    availability: "充足",
    equipment: ["萃茶机"],
  },
  {
    id: "black_tea",
    name: "锡兰红茶汤",
    category: "茶底",
    costPerUnit: "0.016元/ml",
    flavorTags: ["醇厚", "焦糖感", "茶感"],
    allergens: [],
    availability: "充足",
    equipment: ["萃茶机"],
  },
  {
    id: "oolong_tea",
    name: "桂花乌龙茶汤",
    category: "茶底",
    costPerUnit: "0.022元/ml",
    flavorTags: ["桂花香", "焙火", "回甘"],
    allergens: [],
    availability: "季节限定",
    equipment: ["萃茶机"],
  },
  {
    id: "fresh_milk",
    name: "鲜牛奶",
    category: "奶基底",
    costPerUnit: "0.038元/ml",
    flavorTags: ["奶香", "顺滑", "清洁感"],
    allergens: ["乳制品"],
    availability: "充足",
    equipment: ["冷藏柜"],
  },
  {
    id: "light_milk",
    name: "轻乳基底",
    category: "奶基底",
    costPerUnit: "0.031元/ml",
    flavorTags: ["轻盈", "低负担", "柔和"],
    allergens: ["乳制品"],
    availability: "充足",
    equipment: ["冷藏柜"],
  },
  {
    id: "cheese_foam",
    name: "芝士奶盖",
    category: "奶基底",
    costPerUnit: "0.065元/g",
    flavorTags: ["咸甜", "浓郁", "绵密"],
    allergens: ["乳制品"],
    availability: "偏低",
    equipment: ["奶盖机", "冷藏柜"],
  },
  {
    id: "grape_juice",
    name: "青提果汁",
    category: "水果",
    costPerUnit: "0.058元/ml",
    flavorTags: ["青提", "果香", "清爽"],
    allergens: [],
    availability: "充足",
    equipment: ["冷藏柜"],
  },
  {
    id: "mango_puree",
    name: "芒果果泥",
    category: "水果",
    costPerUnit: "0.052元/g",
    flavorTags: ["热带", "甜香", "厚实"],
    allergens: [],
    availability: "充足",
    equipment: ["冷藏柜"],
  },
  {
    id: "strawberry_jam",
    name: "草莓果酱",
    category: "水果",
    costPerUnit: "0.049元/g",
    flavorTags: ["莓果", "酸甜", "明亮"],
    allergens: [],
    availability: "季节限定",
    equipment: ["冷藏柜"],
  },
  {
    id: "coconut_jelly",
    name: "椰果",
    category: "小料",
    costPerUnit: "0.026元/g",
    flavorTags: ["脆弹", "椰香", "清爽"],
    allergens: [],
    availability: "充足",
    equipment: [],
  },
  {
    id: "jasmine_jelly",
    name: "茉莉茶冻",
    category: "小料",
    costPerUnit: "0.032元/g",
    flavorTags: ["茶香", "嫩滑", "清爽"],
    allergens: [],
    availability: "充足",
    equipment: ["冷藏柜"],
  },
  {
    id: "brown_sugar_boba",
    name: "黑糖珍珠",
    category: "小料",
    costPerUnit: "0.035元/g",
    flavorTags: ["黑糖", "嚼感", "厚重"],
    allergens: [],
    availability: "充足",
    equipment: ["煮珍珠锅"],
  },
  {
    id: "cane_syrup",
    name: "蔗糖糖浆",
    category: "风味糖浆",
    costPerUnit: "0.018元/ml",
    flavorTags: ["甜感", "干净", "基础"],
    allergens: [],
    availability: "充足",
    equipment: [],
  },
  {
    id: "osmanthus_syrup",
    name: "桂花糖浆",
    category: "风味糖浆",
    costPerUnit: "0.041元/ml",
    flavorTags: ["花香", "甜润", "东方感"],
    allergens: [],
    availability: "季节限定",
    equipment: [],
  },
  {
    id: "ice",
    name: "冰块",
    category: "辅料",
    costPerUnit: "0.004元/g",
    flavorTags: ["冰爽", "降温"],
    allergens: [],
    availability: "充足",
    equipment: ["制冰机"],
  },
];

export const defaultSelectedIngredientIds = [
  "jasmine_green_tea",
  "light_milk",
  "grape_juice",
  "jasmine_jelly",
  "cane_syrup",
  "ice",
];

export const defaultStoreProfile: StoreProfile = {
  storeName: "AI 奶茶实验店",
  storeType: "商圈快取店",
  brandStyle: "清爽、年轻、低负担",
  equipment: ["萃茶机", "冷藏柜", "制冰机", "封口机"],
};

export const equipmentOptions = [
  "萃茶机",
  "冷藏柜",
  "制冰机",
  "封口机",
  "奶盖机",
  "煮珍珠锅",
];

export type StoreConfig = {
  storeProfile: StoreProfile;
  selectedIngredientIds: string[];
};

export function getDefaultStoreConfig(): StoreConfig {
  return {
    storeProfile: defaultStoreProfile,
    selectedIngredientIds: defaultSelectedIngredientIds,
  };
}

export function normalizeStoreConfig(config: Partial<StoreConfig>): StoreConfig {
  return {
    storeProfile: {
      ...defaultStoreProfile,
      ...(config.storeProfile ?? {}),
      equipment: Array.isArray(config.storeProfile?.equipment)
        ? config.storeProfile.equipment
        : defaultStoreProfile.equipment,
    },
    selectedIngredientIds: Array.isArray(config.selectedIngredientIds)
      ? config.selectedIngredientIds
      : defaultSelectedIngredientIds,
  };
}
