import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { calls, emitLink, frame, mockTauri } from "./tauri-mock.js";

const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");
const plan = JSON.parse(fixture("example-plan.json"));
const guidance = JSON.parse(fixture("example-guidance.json"));
const yaml = fixture("example.yaml");

async function openTimeline(page, answers) {
  await mockTauri(page, answers);
  await page.goto("/");
  await page.getByRole("button", { name: "Timeline" }).click();
}

async function chooseYaml(page, name = "hod-insane.yaml") {
  await page.locator('input[type="file"]').setInputFiles({
    name,
    mimeType: "text/yaml",
    buffer: Buffer.from(yaml),
  });
}

test("a loaded timeline lists its rows and follows the battle", async ({ page }) => {
  await openTimeline(page, { load_timeline: { ok: plan }, select_timeline_row: { ok: true } });
  await expect(page.getByText("Open a timeline YAML made in Timeline Arona.")).toBeVisible();

  await chooseYaml(page);
  expect(await calls(page)).toEqual([{ cmd: "load_timeline", args: { yaml } }]);
  await expect(page.getByText("ホド INSANE · hod-insane.yaml")).toBeVisible();
  const rows = page.locator("tbody tr");
  await expect(rows).toHaveCount(9);
  await expect(rows.nth(0)).toContainText("3:30");
  await expect(rows.nth(0)).toContainText("コタマ");
  await expect(rows.nth(0)).toContainText("開幕バフ");
  await expect(rows.nth(3)).toContainText("コタマNS ボス攻撃力デバフ発動");
  await expect(rows.nth(4)).toContainText("Cost 7.1");
  await expect(rows.nth(8)).toContainText("1:20.500");
  await expect(rows.nth(8)).toContainText("シロコ");

  // The battle has used コタマ: ホシノ (row 1) is next.
  await emitLink(page, frame(guidance));
  await expect(rows.nth(1).getByLabel("next")).toBeVisible();
  await expect(rows.nth(1)).toHaveClass(/font-bold/);
  await expect(rows.nth(0)).toHaveClass(/text-slate-400/);

  // After a misread the Sensei goes on from another skill.
  await rows.nth(5).click();
  expect((await calls(page)).at(-1)).toEqual({ cmd: "select_timeline_row", args: { row: 5 } });
  // A marker is no skill to go on from.
  await rows.nth(3).click();
  expect((await calls(page)).at(-1).args).toEqual({ row: 5 });

  // The timeline stays loaded while another tab is open.
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Timeline" }).click();
  await expect(page.locator("tbody tr")).toHaveCount(9);
});

test("a timeline with mistakes says what is wrong where", async ({ page }) => {
  const errors = [
    "boss がありません",
    "timeline の 2 行目: skill \"ヒナ\" は party にいません",
  ];
  await openTimeline(page, { load_timeline: { error: errors } });
  await chooseYaml(page);
  await expect(page.getByText("This file cannot be used")).toBeVisible();
  for (const error of errors) {
    await expect(page.getByText(error)).toBeVisible();
  }
  await expect(page.locator("tbody tr")).toHaveCount(0);
});

test("the overlay button waits for the game", async ({ page }) => {
  await openTimeline(page, { load_timeline: { ok: plan } });
  const overlay = page.getByRole("button", { name: "Show overlay" });
  await expect(overlay).toBeDisabled();
  await emitLink(page, frame(null));
  await expect(overlay).toBeEnabled();
});

test("the overlay shows the group, the next skill and a marker about to come", async ({ page }) => {
  await mockTauri(page);
  await page.goto("/overlay.html?theme=dark");
  await expect(page.locator(".card")).toHaveCount(0);

  await emitLink(page, frame(guidance));
  const card = page.locator(".card");
  await expect(card).toHaveClass(/due/);
  await expect(card.locator(".trigger")).toHaveText(/NOW\s*3:30/);
  const skills = card.locator(".skill-row");
  await expect(skills).toHaveText(["コタマ", "▶ ホシノ", "シロコ"]);
  await expect(skills.nth(0)).toHaveClass(/done/);
  await expect(card.locator(".mark")).toHaveText("3:25 コタマNS ボス攻撃力デバフ発動");

  // Waiting for the cost, with a target, and the marker far away.
  await emitLink(
    page,
    frame({
      ...guidance,
      next: 8,
      due: false,
      trigger: { kind: "cost", cost: 7.1 },
      skills: [{ row: 8, skill: "セリナ", target: "シロコ", note: "回復", done: false }],
      mark: { ...guidance.mark, soon: false },
    }),
  );
  await expect(card).not.toHaveClass(/due/);
  await expect(card.locator(".trigger")).toHaveText("Cost 7.1");
  await expect(card.locator(".target")).toHaveText("→ シロコ");
  await expect(card.locator(".mark")).toHaveCount(0);

  // Every skill used: nothing to show.
  await emitLink(page, frame({ ...guidance, next: null, skills: [], mark: null }));
  await expect(card).toHaveCount(0);
});
