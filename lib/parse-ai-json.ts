export function parseAIJsonObject<T>(content: string): T {
  const cleaned = stripThinking(content).trim();
  const candidates = [
    cleaned,
    ...extractFencedBlocks(cleaned),
    extractBalancedObject(cleaned),
  ].filter(Boolean);

  for (const candidate of candidates) {
    try {
      return JSON.parse(repairJson(candidate)) as T;
    } catch {
      // Try the next candidate.
    }
  }

  throw new Error("No parseable JSON object found in AI response.");
}

function stripThinking(content: string) {
  return content
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/^\uFEFF/, "");
}

function extractFencedBlocks(content: string) {
  return Array.from(content.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi)).map(
    (match) => match[1].trim(),
  );
}

function extractBalancedObject(content: string) {
  let start = -1;
  let depth = 0;
  let inString = false;
  let escaping = false;

  for (let index = 0; index < content.length; index += 1) {
    const char = content[index];

    if (start < 0) {
      if (char === "{") {
        start = index;
        depth = 1;
      }
      continue;
    }

    if (escaping) {
      escaping = false;
      continue;
    }

    if (char === "\\") {
      escaping = inString;
      continue;
    }

    if (char === "\"") {
      inString = !inString;
      continue;
    }

    if (inString) {
      continue;
    }

    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return content.slice(start, index + 1);
      }
    }
  }

  return "";
}

function repairJson(content: string) {
  return content
    .trim()
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/i, "")
    .replace(/,\s*([}\]])/g, "$1")
    .trim();
}
