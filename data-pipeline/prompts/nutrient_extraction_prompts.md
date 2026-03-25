# Nutrient Extraction Prompts (Supplier PDFs)

This document records the exact prompts and field definitions used in the automated n8n data collection pipeline to extract nutritional information from supplier ingredient specification PDFs using GPT-5.1 (OpenAI API). This pipeline collected the 12-nutrient composition data for 28 ingredients used to compute NRF9.3 scores in the paper.

---

## Overview

Supplier specification PDFs (sourced via TraceGains) were processed by an automated n8n workflow (see `../n8n/GoogleDrive-Mistral-GeminiChat-GoogleSheets_v3.json`), which:

1. Detected new PDFs in a designated Google Drive input folder via a schedule trigger (every 2 minutes).
2. Extracted raw text from each PDF using Mistral AI.
3. Passed the extracted text to a **GPT-5.1 Information Extractor** with the system prompt and field schema below.
4. Wrote the structured output to a Google Sheet (column schema documented at the end of this file).
5. Moved processed files to an archive folder and renamed them as `{ingredient_category} - {product_id}`.

---

## System Prompt

```
You are an expert extraction algorithm. You are also an expert in nutrition, so you understand synonyms of them (e.g. fiber, dietary fiber, crude fiber could be considered as the same thing).
Only extract relevant information from the text.
If you do not know the value of an attribute asked to extract, you may omit the attribute's value. Never put 0 unless it is being provided in the text.
IMPORTANT: Return ONLY valid JSON without any markdown formatting, code blocks, or backticks. Do not wrap the JSON in ```json or ``` tags.
```

---

## Extraction Field Schema

Each field below was passed to the LLM as a structured attribute with a name, description, type, and required flag.

### Product & Supplier Metadata

| Field | Type | Required | Description |
|---|---|---|---|
| `ingredient_category` | string | yes | Select from predefined categories only — do not create new categories. Allowed values: Corn flour, Wheat flour, Rice flour, Oat flour, Barley flour, Rye flour, Millet flour, Buckwheat flour, Sorghum flour, Peanut flour, Almond flour, Pea protein, Chickpea, Lentil, Mung bean, Pea flour, Brown rice flour, Rice protein, Pea fiber, Tomato powder, Garlic powder, Tapioca starch, Onion powder, Canola oil, Sunflower oil, Palm oil, Strawberry powder, Apple powder. |
| `supplier_id` | string | yes | Supplier's identifier or code used in the file. Treat as text even if it looks numeric. |
| `product_id` | string | yes | Unique identifier assigned to the ingredient product in the file (e.g., Supplier Item ID, SKU, Product Code). Usually a sequence of numbers or letters. |
| `product_name` | string | yes | Full product name as listed in the file (e.g., "Organic Whole Wheat Flour"). |
| `supplier_name` | string | yes | Name of the supplier providing the ingredient, as shown in the document. Extract as written in the Nutrition PDF. |
| `description` | string | yes | Descriptive text about the product, such as composition, usage, or other notes, as written in the document. |
| `location` | string | yes | Location related to the product or supplier (e.g., manufacturing site, warehouse, city, or country) as stated in the document. |

### Nutritional Values

For each nutrient, two fields are extracted: a numeric `_value` (without units) and the corresponding `_unit` string exactly as written in the source document.

| Nutrient | Value field | Unit field | Notes |
|---|---|---|---|
| Protein | `protein_value` | `protein_unit` | e.g., `"g"`, `"mg"` |
| Total dietary fiber | `fiber_total_dietary_value` | `fiber_total_dietary_unit` | Synonyms: fiber, dietary fiber, crude fiber |
| Vitamin A (RAE) | `vitamin_a_rae_value` | `vitamin_a_rae_unit` | e.g., `"µg"`, `"mcg"`, `"IU"` |
| Vitamin C | `vitamin_c_value` | `vitamin_c_unit` | Total ascorbic acid |
| Vitamin E | `vitamin_e_value` | `vitamin_e_unit` | Alpha-tocopherol |
| Calcium | `calcium_value` | `calcium_unit` | |
| Iron | `iron_value` | `iron_unit` | |
| Magnesium | `magnesium_value` | `magnesium_unit` | |
| Potassium | `potassium_value` | `potassium_unit` | |
| Saturated fatty acids | `saturated_fatty_acids_value` | `saturated_fatty_acids_unit` | Total saturated fatty acids |
| Added sugars | `sugars_added_value` | `sugars_added_unit` | |
| Sodium | `sodium_value` | `sodium_unit` | |
| Energy | `energy_combined_value` | `energy_combined_unit` | e.g., `"kcal"`, `"cal"` |

> **Extraction rule for numeric values:** Omit the field entirely if the nutrient is not found in the document. Never insert `0` unless the value is explicitly stated as `0` in the source. A `null` or missing value is preserved as an empty string in the final output (handled by a post-processing code node).

---

## LLM Configuration

| Parameter | Value |
|---|---|
| Model (extraction) | GPT-5.1 (`gpt-5.1`) via OpenAI API |
| Model (text extraction from PDF) | Mistral AI |
| n8n node type | `@n8n/n8n-nodes-langchain.informationExtractor` v1.2 |
| Temperature | Default (not overridden) |

---

## Post-Processing Logic

After LLM extraction, a JavaScript code node (`Handle Zero vs Empty Values`) converts `null` or `undefined` numeric fields to empty strings while preserving explicit `0` values:

```javascript
const numericFields = [
  'protein_value', 'fiber_total_dietary_value',
  'vitamin_a_rae_value', 'vitamin_c_value', 'vitamin_e_value',
  'calcium_value', 'iron_value', 'magnesium_value',
  'potassium_value', 'saturated_fatty_acids_value',
  'sugars_added_value', 'sodium_value', 'energy_combined_value'
];

for (const field of numericFields) {
  if (output[field] === null || output[field] === undefined) {
    output[field] = '';
  }
  // Explicit 0 values are kept as-is
}
```

---

## Output Schema (Google Sheets columns)

| Column | Source field |
|---|---|
| Ingredient Category | `ingredient_category` |
| Product ID | `product_id` |
| File Name | Original filename |
| File URL | Google Drive file URL |
| Product Name | `product_name` |
| Supplier Name | `supplier_name` |
| Supplier ID | `supplier_id` |
| Description | `description` |
| Location | `location` |
| Protein (g) | `protein_value` |
| Fiber, total dietary (g) | `fiber_total_dietary_value` |
| Calcium (mg) | `calcium_value` |
| Iron (mg) | `iron_value` |
| Potassium (mg) | `potassium_value` |
| Saturated Fatty Acids (g) | `saturated_fatty_acids_value` |
| Added Sugars (g) | `sugars_added_value` |
| Energy (kcal) | `energy_combined_value` |
| Sodium (mg) | `sodium_value` |
| Vitamin A (IU; mcg RAE) | `vitamin_a_rae_value` |
| Vitamin C (mg) | `vitamin_c_value` |
| Vitamin E (mg) | `vitamin_e_value` |
| Magnesium (mg) | `magnesium_value` |
