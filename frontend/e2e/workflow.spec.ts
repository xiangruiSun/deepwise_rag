import { expect, test } from "@playwright/test";

test.describe("research workflow on the synthetic cohort", () => {
  test("search page loads with the research-only notice and supported filters", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("note")).toContainText("Research use only");
    await expect(page.getByText("SYNTHETIC DATA").first()).toBeVisible();
    await expect(page.getByRole("heading", { name: "Similar-case search" })).toBeVisible();
    await expect(page.getByTestId("query-case")).not.toHaveValue("");
    // fields the dataset does not provide are not offered as filters
    await expect(page.getByText(/zip|residence|geograph/i)).toHaveCount(0);
  });

  test("apply filters, choose a retrieval strategy, search, and inspect evidence", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Free text" }).click();
    await page.getByTestId("query-text").fill("acute pulmonary embolism without right heart strain");
    await page.getByTestId("age-min").fill("50");
    await page.getByTestId("age-max").fill("75");
    await page.getByTestId("recipe").selectOption("hybrid_rrf_hashing");
    await page.getByTestId("search-button").click();

    const cards = page.getByTestId("result-card");
    await expect(cards.first()).toBeVisible();
    expect(await cards.count()).toBeGreaterThan(1);
    await expect(page.getByLabel("Search summary")).toContainText("hybrid_rrf_hashing");
    // every returned case satisfies the age filter (filters are applied before retrieval)
    const demographics = await cards.getByTestId("demographics").allTextContents();
    expect(demographics.length).toBeGreaterThan(1);
    for (const text of demographics) {
      const age = Number(text.match(/^(\d+(?:\.\d+)?) y/)?.[1]);
      expect(age).toBeGreaterThanOrEqual(50);
      expect(age).toBeLessThanOrEqual(75);
    }
    // "Why retrieved?" is built from real retrieval evidence
    await cards.first().getByText("Why retrieved?").click();
    await expect(cards.first().locator("details li").first()).toContainText(/rank/);

    // open a result: the source reference leads to the highlighted report
    await cards.first().getByRole("link", { name: /radiology report/ }).click();
    await expect(page).toHaveURL(/\/cases\/case_[0-9a-f]+#source-/);
    await expect(page.getByTestId("report").first()).toBeVisible();
    await expect(page.getByRole("heading", { name: /Follow-up information/ })).toBeVisible();
  });

  test("case search, grounded answer with citations, and side-by-side comparison", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("recipe").selectOption("multimodal_hashing");
    await page.getByTestId("search-button").click();
    const cards = page.getByTestId("result-card");
    await expect(cards.first()).toBeVisible();
    await expect(cards.first().getByLabel("Component relevance indicators")).toContainText("laboratory");

    await page.getByTestId("answer-button").click();
    const answer = page.getByTestId("rag-answer");
    await expect(answer).toBeVisible();
    await expect(answer.getByRole("heading", { name: "Summary" })).toBeVisible();
    await expect(answer.locator("button.cite").first()).toHaveText(/^S\d+$/);

    await cards.nth(0).getByLabel("Compare").check();
    await cards.nth(1).getByLabel("Compare").check();
    await page.getByRole("link", { name: /Compare 2 selected/ }).click();
    await expect(page).toHaveURL(/\/compare\?ids=/);
    await expect(page.getByTestId("compare-column")).toHaveCount(2);
    await expect(page.getByRole("heading", { name: /Laboratory measurements/ })).toBeVisible();
    await expect(page.getByText(/Follow-up after the procedure/).first()).toBeVisible();

    // clicking a citation navigates to the cited source record
    await page.goBack();
    await page.getByTestId("search-button").click();
    await page.getByTestId("answer-button").click();
    await page.getByTestId("rag-answer").locator("button.cite").first().click();
    await expect(page).toHaveURL(/\/cases\/case_[0-9a-f]+#source-/);
  });

  test("an impossible filter returns no case and says so", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Filters only" }).click();
    await page.getByTestId("age-min").fill("125");
    await page.getByTestId("search-button").click();
    await expect(page.getByTestId("eligible-count")).toHaveText("0");
    await expect(page.getByText("No authorized cases satisfy the requested filters.")).toBeVisible();
  });

  test("evaluation dashboard shows recorded experiments only", async ({ page }) => {
    await page.goto("/evaluation");
    await expect(page.getByRole("heading", { name: "Evaluation dashboard" })).toBeVisible();
    await expect(page.getByText(/SYNTHETIC DATA — not clinical evidence/)).toBeVisible();
    // other suites may have been recorded too; look at the baseline suite explicitly
    await page.getByTestId("suite-select").selectOption("smoke");
    const table = page.getByTestId("experiment-table");
    await expect(table.getByRole("button", { name: "S01" })).toBeVisible();
    await expect(page.getByTestId("metric-chart")).toBeVisible();
    await table.getByRole("button", { name: "S06" }).click();
    const detail = page.getByTestId("experiment-detail");
    await expect(detail).toContainText("split isolation");
    await expect(detail).toContainText("multimodal_hashing");
  });
});
