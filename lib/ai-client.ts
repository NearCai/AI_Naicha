import OpenAI from "openai";

export function createAIClient() {
  const apiKey = process.env.DEEPSEEK_API_KEY || process.env.AI_API_KEY;
  const baseURL =
    process.env.DEEPSEEK_BASE_URL ||
    process.env.AI_BASE_URL ||
    "https://api.deepseek.com";

  if (!apiKey) {
    return null;
  }

  return new OpenAI({
    apiKey,
    baseURL: baseURL || undefined,
  });
}

export function createImageClient() {
  const apiKey = process.env.IMAGE_API_KEY;
  const baseURL = process.env.IMAGE_BASE_URL;

  if (!apiKey) {
    return null;
  }

  return new OpenAI({
    apiKey,
    baseURL: baseURL || undefined,
  });
}

export function getAIModel() {
  return process.env.DEEPSEEK_MODEL || process.env.AI_MODEL || "deepseek-v4-pro";
}

export function getImageModel() {
  return process.env.IMAGE_MODEL || "doubao-seedream-5-0-260128";
}
