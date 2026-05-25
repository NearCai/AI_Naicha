import { readFile } from "node:fs/promises";
import path from "node:path";
import { HistoryView, type HistoryRecord } from "./history-view";

export const dynamic = "force-dynamic";

async function loadHistory() {
  const historyPath = path.join(process.cwd(), "data", "recipe-history.json");

  try {
    const parsed = JSON.parse(await readFile(historyPath, "utf-8")) as {
      records?: HistoryRecord[];
    };

    return Array.isArray(parsed.records) ? parsed.records : [];
  } catch {
    return [];
  }
}

export default async function HistoryPage() {
  const records = await loadHistory();

  return <HistoryView records={records} />;
}
