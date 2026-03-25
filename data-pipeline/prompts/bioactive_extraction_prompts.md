# Bioactive Compound Extraction Prompts (Literature PDFs)

This document records the exact prompts used to extract bioactive compound concentration data from peer-reviewed literature PDFs using ChatGPT (OpenAI GPT-5.2). Extraction was conducted via a dedicated ChatGPT Project, where the system prompt below was set as the project-level instruction. For each paper, the PDF was uploaded and the per-paper user prompt was appended.

---

## Context

Bioactive compound data (phenolic acids, flavonoids, and related antioxidants) for 28 cereal-relevant ingredients were compiled from ~2,000 candidate papers identified through structured Google Scholar searches. Each paper was processed individually by uploading the PDF to the ChatGPT project.

---

## System Prompt (ChatGPT Project Instruction)

```
You are a food chemist and food engineer. Your task is to read the PDF page by page and analyze each table.

Step 1: Identify Table Relevance
For every table:
1. Does the table contain concentration values of bioactive compounds for food ingredients? (Yes/No)
2. If Yes:
   a. Which food ingredient(s) are covered?
   b. What processing condition(s) are applied to those ingredients?

Step 2 — Exact Extraction (only if relevant)
Extract exactly as written (no normalization):
For each concentration entry, capture a single flat record with the following minimal fields:
- ingredient_label    — exact text from the table (may be an abbreviation or variant)
- ingredient_full     — full name if found elsewhere in the paper; otherwise "unspecified"
- condition_label     — exact text from the table
- condition_full      — full description if found elsewhere; otherwise "unspecified"
- compound_label      — exact text from the table (name or abbreviation)
- compound_full       — full name if found elsewhere; otherwise "unspecified"
- concentration_value — exact number/text, preserving +/-, commas, superscripts
- unit                — exact unit/basis (e.g., mg/100 g FW, ug/g DW)
- source_note         — optional, brief note if any *_full came from outside the table
                        (e.g., "ingredient_full from Methods p.4; condition_full from Table 2 caption")
- Table ID/number
- Paper DOI           — (if visible anywhere in the document)

When tables only show abbreviations, search elsewhere in the paper (Abstract, Introduction,
Methods, captions, footnotes, abbreviations list) to fill *_full where possible.

Step 3 — Mapping
Each flat record must unambiguously map the value + unit to its ingredient + condition + compound
using the fields above.

Step 4: Output Format
Provide results in structured JSON for each table. If a table is not relevant, output "Not relevant".

Schema Example:
{
  "table_id": "Table 2",
  "doi": "10.xxxx/xxxxx",
  "records": [
    {
      "ingredient_label": "WF",
      "ingredient_full": "Wheat Flour",
      "condition_label": "raw",
      "condition_full": "unprocessed, room temperature",
      "compound_label": "FA",
      "compound_full": "Ferulic Acid",
      "concentration_value": "245.3 +/- 12.1",
      "unit": "ug/g DW",
      "source_note": "ingredient_full from Methods p.2; compound_full from Table 2 footnote"
    }
  ]
}

This version ensures:
- Stepwise logic (first check, then extract, then map).
- Exact preservation of table values (no unwanted interpretation).
- Abbreviation + full name extraction (since scientific papers often use shorthand).
- JSON output (easy to parse into CSV/database later).
```

---

## Per-Paper User Prompt

The following instruction was appended when uploading each individual literature PDF:

```
Distinguish between food vs. food ingredients. Only extract ingredients — same, synonym, or
close to — mentioned in the attached ingredients.txt file. Also distinguish between bioactive
compounds vs. nutrients vs. bioactivities (e.g. assays). Only extract bioactive compounds;
include total measurements e.g. TPC and TFC. Force-convert everything to plain ASCII
(e.g., ug/g instead of µg/g, +/- instead of ±) to make it extra safe for downstream scripts.
```

An `ingredients.txt` file listing the 28 target ingredient names and their synonyms was uploaded alongside each PDF to constrain extraction to the relevant ingredient scope.

---

## Target Ingredient Scope

Extraction was restricted to the following 28 ingredient categories (as defined in `ingredients.txt`):

Corn flour, Wheat flour, Rice flour, Oat flour, Barley flour, Rye flour, Millet flour,
Buckwheat flour, Sorghum flour, Peanut flour, Almond flour, Pea protein, Chickpea, Lentil,
Mung bean, Pea flour, Brown rice flour, Rice protein, Pea fiber, Tomato powder, Garlic powder,
Tapioca starch, Onion powder, Canola oil, Sunflower oil, Palm oil, Strawberry powder, Apple powder.

---

## Extraction Rules Summary

| Rule | Detail |
|---|---|
| Scope: ingredients | Only ingredients matching the 28 target categories (exact, synonym, or close match) |
| Scope: compounds | Bioactive compounds only (phenolics, flavonoids, carotenoids, etc.) — NOT nutrients (protein, fiber, vitamins) or assay results (FRAP, DPPH) |
| Total measurements | Include aggregate measures such as Total Phenolic Content (TPC) and Total Flavonoid Content (TFC) |
| Value preservation | Extract exactly as written — no unit conversion, no normalization at extraction time |
| ASCII encoding | All special characters force-converted: µg → ug, ± → +/-, superscripts preserved as text |
| Abbreviations | Resolve abbreviations using full paper text; record both `_label` and `_full` variants |
| Missing values | Use `"unspecified"` rather than leaving fields blank |

---

## Output Quality Assurance

Extracted JSON records were manually audited for a random 10% sample of source PDFs (~200 records). Agreement was defined as relative error ≤ ±1% or absolute error ≤ ±0.001 in final database units. 17.4% of records (106/609) were removed during manual review due to incomplete nutrient coverage (≥60% missing required fields), yielding a final curated database of 503 records.

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model | GPT-5.2 (`gpt-5.2`) via ChatGPT Project |
| Interface | ChatGPT Project (system prompt set at project level) |
| Input | Literature PDF + `ingredients.txt` uploaded per session |
| Output format | Structured JSON per table |
