"""Beauty-domain prompt packs, one per expansion variant.

Prior, scoring and explanation prompts are identical across the standard, CoT-SC and ToT
variants, so they are defined once here; only the expansion prompts differ. Templates use the
pipeline's placeholder names: `{text_claim}` for the advertised techniques,
`{retrieved_regulatory_context}` for the registered business types and `{verdict}` for the
adjudicated verdict. The ToT prompts additionally reference `{state}`.

The `\"\"\"` delimiters around `{state}` are prompt text the model sees, not Python syntax.
"""

from __future__ import annotations

from mmregcheck.pillar2.prompts._pack import ExpansionName, PromptPack
from mmregcheck.pillar2.prompts._tot_schemas import STATE_EVAL_SCHEMA, THOUGHT_GEN_SCHEMA
from mmregcheck.schemas import RegulatoryDomain

# Stage 1 — Prior: only `{text_claim}` is available (no evidence yet).
_PRIOR_PROMPT = """\
You are a compliance classifier. You output ONLY a single digit. Nothing else.

A beauty clinic in Vietnam advertises these techniques:
[Techniques]: {text_claim}

Classification rules (apply in priority order):
  - Any technique is Non-Compliant  --> output 0
  - Any technique is Partly-Compliant --> output 1
  - All techniques are Compliant    --> output 2

Output exactly one digit (0, 1, or 2). Do not output any text, punctuation, explanation, or whitespace before or after the digit.
"""

# Stage 2 — Standard expansion.
_STANDARD_EXPANSION_PROMPT = """\
You are an expert in Vietnamese beauty regulations, specializing in verifying whether the techniques or services advertised by a beauty facility comply with its registered business types (e.g., SPA, massage, hair salon, tattooing/micropigmentation).

You will be provided with:
- **[Allowed_Services]** — the registered business types of the establishment
- **[Techniques]** — techniques or services found in the advertisement

Based on the regulations and business types of the establishment below, determine whether the establishment is permitted to carry out the listed techniques (multiple business types may be combined — for example, if an establishment is both a SPA and a tattooing/micropigmentation facility, it is permitted to perform both skincare and micropigmentation services).

**Regulations:**

1. **SPA establishments and Skincare establishments** may NOT perform aesthetic services that use drugs, substances, or devices to intervene in the human body (such as surgery, procedures, or interventions involving injection, infusion, pumping, radiation, waves, cauterization, or other interventions) that alter skin color, body shape, weight, or defects of body parts, or tattooing/micropigmentation using injectable anesthetics.

2. **Massage establishments** may use physical therapy methods for massage and body treatment for human health purposes.

3. **Hair salon establishments and Hair-washing establishments** may perform:
   - 3.1. Cutting and styling: Men's and women's haircuts, styling (perming, straightening, dyeing, hairstyling). Professional hair care (conditioning, hair restoration).
   - 3.2. Hair washing and care: Regular hair washing or washing combined with scalp massage; Scalp care (dandruff removal, scalp treatment).
   - 3.3. Basic appearance care (if additionally registered): Eyebrow trimming, shaving. Nail care (manicure, pedicure). Facial massage (if non-invasive devices are used); Simple on-site steam services.

4. **Tattooing/Micropigmentation establishments** may perform:
   - 4.1. Cosmetic micropigmentation: Shaping and correcting eyebrows, lips, and eyelids using micropigmentation techniques.
   - 4.2. Cosmetic embroidery: Using embroidery techniques to create natural-looking eyebrows.
   - 4.3. Artistic tattooing: Performing body tattoos on any body part per customer request.

5. **Note:** None of the above establishments may use injectable anesthetics during service delivery.

---

[Allowed_Services]: {retrieved_regulatory_context}

[Techniques]: {text_claim}

---

## Output Format

Return a concise reasoning paragraph analyzing the techniques. NO YAPPING.
"""

# Stage 2 — CoT-SC expansion.
_COTSC_EXPANSION_PROMPT = """\
You are an expert in Vietnamese beauty regulations, specializing in verifying whether the techniques or services advertised by a beauty facility comply with its registered business types (e.g., SPA, massage, hair salon, tattooing/micropigmentation).

You will be provided with:
- **[Allowed_Services]** — the registered business types of the establishment
- **[Techniques]** — techniques or services found in the advertisement

Based on the regulations and business types of the establishment below, determine whether the establishment is permitted to carry out the listed techniques (multiple business types may be combined — for example, if an establishment is both a SPA and a tattooing/micropigmentation facility, it is permitted to perform both skincare and micropigmentation services).

**Regulations:**

1. **SPA establishments and Skincare establishments** may NOT perform aesthetic services that use drugs, substances, or devices to intervene in the human body (such as surgery, procedures, or interventions involving injection, infusion, pumping, radiation, waves, cauterization, or other interventions) that alter skin color, body shape, weight, or defects of body parts, or tattooing/micropigmentation using injectable anesthetics.

2. **Massage establishments** may use physical therapy methods for massage and body treatment for human health purposes.

3. **Hair salon establishments and Hair-washing establishments** may perform:
   - 3.1. Cutting and styling: Men's and women's haircuts, styling (perming, straightening, dyeing, hairstyling). Professional hair care (conditioning, hair restoration).
   - 3.2. Hair washing and care: Regular hair washing or washing combined with scalp massage; Scalp care (dandruff removal, scalp treatment).
   - 3.3. Basic appearance care (if additionally registered): Eyebrow trimming, shaving. Nail care (manicure, pedicure). Facial massage (if non-invasive devices are used); Simple on-site steam services.

4. **Tattooing/Micropigmentation establishments** may perform:
   - 4.1. Cosmetic micropigmentation: Shaping and correcting eyebrows, lips, and eyelids using micropigmentation techniques.
   - 4.2. Cosmetic embroidery: Using embroidery techniques to create natural-looking eyebrows.
   - 4.3. Artistic tattooing: Performing body tattoos on any body part per customer request.

5. **Note:** None of the above establishments may use injectable anesthetics during service delivery.

---

[Allowed_Services]: {retrieved_regulatory_context}

[Techniques]: {text_claim}

---

## Output Format

Let's think step by step to analyze the compliance:
1. Interpret [Allowed_Services] to directly determine the explicit regulatory category of the facility (e.g., SPA/Skincare, Massage, Hair Salon, Tattooing/Micropigmentation).
2. STRICT BOUNDARIES: A facility is strictly bound by its registered category. It CANNOT perform techniques from a different regulatory category unless that category is explicitly listed in [Allowed_Services].
3. Analyse the mechanics of the [Techniques] to determine whether they fall within the permitted scope. Do not assume cross-domain compliance based on industry customs.
4. Finally, produce a clean, concise, step-by-step analytical reasoning chain that maps each technique in [Techniques] to the relevant regulation rule and the facility's registered category.

Do not output the final score digit here, just the reasoning process. Reason through ALL techniques; partial analysis is invalid.
"""

# Stage 2 — ToT thought generation (the JSON response schema is appended verbatim).
_TOT_THOUGHT_GEN_PROMPT = (
    """\
You are an expert in Vietnamese beauty regulations, specializing in verifying whether
the techniques or services advertised by a beauty facility comply with its registered
business types (e.g., SPA, massage, hair salon, tattooing/micropigmentation).

**Regulations:**

1. **SPA establishments and Skincare establishments** may NOT perform aesthetic services
   that use drugs, substances, or devices to intervene in the human body (such as surgery,
   procedures, or interventions involving injection, infusion, pumping, radiation, waves,
   cauterization, or other interventions) that alter skin color, body shape, weight, or
   defects of body parts, or tattooing/micropigmentation using injectable anesthetics.

2. **Massage establishments** may use physical therapy methods for massage and body
   treatment for human health purposes.

3. **Hair salon establishments and Hair-washing establishments** may perform:
   - 3.1. Cutting and styling: Men's and women's haircuts, styling (perming, straightening,
     dyeing, hairstyling). Professional hair care (conditioning, hair restoration).
   - 3.2. Hair washing and care: Regular hair washing or washing combined with scalp massage;
     Scalp care (dandruff removal, scalp treatment).
   - 3.3. Basic appearance care (if additionally registered): Eyebrow trimming, shaving.
     Nail care (manicure, pedicure). Facial massage (if non-invasive devices are used);
     Simple on-site steam services.

4. **Tattooing/Micropigmentation establishments** may perform:
   - 4.1. Cosmetic micropigmentation: Shaping and correcting eyebrows, lips, and eyelids
     using micropigmentation techniques.
   - 4.2. Cosmetic embroidery: Using embroidery techniques to create natural-looking eyebrows.
   - 4.3. Artistic tattooing: Performing body tattoos on any body part per customer request.

5. **Note:** None of the above establishments may use injectable anesthetics during service
   delivery.

---

[Allowed_Services]: {retrieved_regulatory_context}
[Techniques]: {text_claim}

Reasoning so far:
{state}

---

**Reasoning Framework:**
  1. Interpret [Allowed_Services] to directly determine the explicit regulatory category of the facility (e.g., SPA/Skincare, Massage, Hair Salon, Tattooing/Micropigmentation).
  2. STRICT BOUNDARIES: A facility is strictly bound by its registered category. It CANNOT perform techniques from a different regulatory category unless that category is explicitly listed in [Allowed_Services].
  3. Analyse the mechanics of the [Techniques] to determine whether they fall within the permitted scope. Do not assume cross-domain compliance based on industry customs.

Continue the analysis based on the reasoning framework above. If all techniques have been analyzed and a conclusion is clear, set terminate=true and provide a final clean and concise reasoning chain that maps each technique in [Techniques] to the relevant regulation rule and the facility's registered category.


"""
    + THOUGHT_GEN_SCHEMA
)

# Stage 2 — ToT branch evaluation (the JSON response schema is appended verbatim).
_TOT_EVAL_PROMPT = (
    """\
You are a compliance auditor reviewing a reasoning branch for Vietnamese beauty regulations.

[Allowed_Services]: {retrieved_regulatory_context}
[Techniques]: {text_claim}

Reasoning branch:
{state}

---

**Evaluation Criteria:**

1. **Category Identification**
   - Correctly determine the facility category from [Allowed_Services].
   - Do not assume additional categories.

2. **Strict Boundaries**
   - Only allow services within the registered category.
   - Any cross-category reasoning = critical error.

3. **Technique Coverage**
   - All techniques must be analyzed individually.
   - Must assess mechanism (e.g., injection, laser, invasive), not just names.

4. **Regulation Mapping**
   - Apply rules correctly:
     - SPA/Skincare → no invasive/body-altering procedures
     - Massage → physical therapy only
     - Hair salon → hair + basic non-invasive care
     - Tattooing → pigment only, no injectable anesthetics

5. **Prohibited Actions**
   - Injections, anesthetics, lasers, surgery, or similar interventions must be flagged as NOT allowed.

6. **Consistency**
   - No missing techniques or contradictions.
   - Leads to a regulation-consistent conclusion.

---

**Rating:**
- "sure" → Fully correct, complete, and consistent\x20\x20
- "likely" → Mostly correct, minor gaps\x20\x20
- "impossible" → Any critical error (wrong category, boundary violation, misclassification, or missing technique)

---

Return evaluation per schema.
"""
    + STATE_EVAL_SCHEMA
)

# Stage 3 — Scoring.
_SCORING_PROMPT = """\
You are a compliance classifier. You output ONLY a single digit. Nothing else.

[Allowed Services]: {retrieved_regulatory_context}
[Techniques]: {text_claim}

Based on the regulations and business types of the establishment below, determine whether the establishment is permitted to carry out the listed techniques (multiple business types may be combined — for example, if an establishment is both a SPA and a tattooing/micropigmentation facility, it is permitted to perform both skincare and micropigmentation services).

**Regulations:**

1. **SPA establishments and Skincare establishments** may NOT perform aesthetic services that use drugs, substances, or devices to intervene in the human body (such as surgery, procedures, or interventions involving injection, infusion, pumping, radiation, waves, cauterization, or other interventions) that alter skin color, body shape, weight, or defects of body parts, or tattooing/micropigmentation using injectable anesthetics.

2. **Massage establishments** may use physical therapy methods for massage and body treatment for human health purposes.

3. **Hair salon establishments and Hair-washing establishments** may perform:
   - 3.1. Cutting and styling: Men's and women's haircuts, styling (perming, straightening, dyeing, hairstyling). Professional hair care (conditioning, hair restoration).
   - 3.2. Hair washing and care: Regular hair washing or washing combined with scalp massage; Scalp care (dandruff removal, scalp treatment).
   - 3.3. Basic appearance care (if additionally registered): Eyebrow trimming, shaving. Nail care (manicure, pedicure). Facial massage (if non-invasive devices are used); Simple on-site steam services.

4. **Tattooing/Micropigmentation establishments** may perform:
   - 4.1. Cosmetic micropigmentation: Shaping and correcting eyebrows, lips, and eyelids using micropigmentation techniques.
   - 4.2. Cosmetic embroidery: Using embroidery techniques to create natural-looking eyebrows.
   - 4.3. Artistic tattooing: Performing body tattoos on any body part per customer request.

5. **Note:** None of the above establishments may use injectable anesthetics during service delivery.

Reasoning path support your task:
\"\"\"
{state}
\"\"\"

Classification rules (apply in priority order):
  - Any technique is Non-Compliant   --> output 0
  - Any technique is Partly-Compliant --> output 1
  - All techniques are Compliant     --> output 2

Output exactly one digit (0, 1, or 2). No text, no punctuation, no whitespace.

Final Answer (0, 1, or 2):"""

# Stage 5 — Explanation.
_EXPLANATION_PROMPT = """\
You are a fact-checking auditor.

Evidence: {retrieved_regulatory_context}
Claim: {text_claim}
Verdict: {verdict}

Supporting reasoning:
{paths}

Write 2-3 sentences explaining why the claim is {verdict} in Vietnamese. Do not mention variable names."""


BEAUTY_PACKS: dict[ExpansionName, PromptPack] = {
    "standard": PromptPack(
        domain=RegulatoryDomain.BEAUTY,
        expansion="standard",
        prior=_PRIOR_PROMPT,
        expansion_prompt=_STANDARD_EXPANSION_PROMPT,
        scoring=_SCORING_PROMPT,
        explanation=_EXPLANATION_PROMPT,
        n_paths=3,
        expansion_max_tokens=1024,
        expansion_temperature=0.2,
    ),
    "cot-sc": PromptPack(
        domain=RegulatoryDomain.BEAUTY,
        expansion="cot-sc",
        prior=_PRIOR_PROMPT,
        expansion_prompt=_COTSC_EXPANSION_PROMPT,
        scoring=_SCORING_PROMPT,
        explanation=_EXPLANATION_PROMPT,
        n_paths=3,
        expansion_max_tokens=1024,
        expansion_temperature=0.2,
    ),
    "tot": PromptPack(
        domain=RegulatoryDomain.BEAUTY,
        expansion="tot",
        prior=_PRIOR_PROMPT,
        expansion_prompt=_TOT_THOUGHT_GEN_PROMPT,
        scoring=_SCORING_PROMPT,
        explanation=_EXPLANATION_PROMPT,
        tot_eval=_TOT_EVAL_PROMPT,
        tot_max_tokens_thought_gen=1024,
        tot_n_steps=3,
        tot_k=3,
        tot_n_evals=2,
        tot_breadth_limit=3,
        tot_min_steps=2,
    ),
}
