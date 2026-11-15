"""Healthcare-domain prompt packs, one per expansion variant.

Prior, scoring and explanation prompts are identical across the standard, CoT-SC and ToT
variants, so they are defined once here; only the expansion prompts differ. Every expansion and
scoring prompt carries the glossary below, which maps trade names of procedures onto the
regulated technique categories. Templates use the pipeline's placeholder names: `{text_claim}`
for the advertised techniques, `{retrieved_regulatory_context}` for the allowed techniques and
`{verdict}` for the adjudicated verdict. The ToT prompts additionally reference `{state}`.

The `\"\"\"` delimiters around `{state}` are prompt text the model sees, not Python syntax.
"""

from __future__ import annotations

from mmregcheck.pillar2.prompts._pack import ExpansionName, PromptPack
from mmregcheck.pillar2.prompts._tot_schemas import STATE_EVAL_SCHEMA, THOUGHT_GEN_SCHEMA
from mmregcheck.schemas import RegulatoryDomain

# Glossary: static mapping of trade names onto regulated technique categories.
_GLOSSARY = """\
- Nâng mũi cấu trúc dựng trụ Nano Plus, nâng mũi dựng trụ cấu trúc, nâng mũi dựng trụ, nâng mũi Sline, softline → Nâng mũi cấu trúc
- Nâng mũi Surgiform, nâng mũi Megaform → Nâng mũi bằng vật liệu độn nhân tạo
- Nâng mũi sụn tai, sụn tai 2 lớp → Nâng mũi bằng sụn tự thân (kết hợp hoặc không với vật liệu nhân tạo)
- Nâng mũi bằng chỉ collagen → Nâng mũi bằng chỉ sinh học
- Mũi BIO-STICK, nâng mũi với công nghệ không chỉ khâu → Tạo hình mũi
- Cắt mí mắt ILID, ILUX, cắt mí; cắt mí Plasma Deep, cắt mí → Phẫu thuật cắt mí mắt / Tạo hình mắt hai mí
- Nhấn mí baby, nhấn mí bằng chỉ collagen, NẾP MÍ BIO-STICK → Nhấn mí / Tạo hình mí mắt không phẫu thuật
- Cắt mí mở góc mắt → Cắt mí kết hợp Mở rộng góc mắt
- Nâng cao nếp mí → Tạo hình mí mắt (Cắt mí hoặc Nhấn mí)
- Độn cằm Surgiform, Megaderm, Megaform, đặt cằm → Phẫu thuật độn cằm
- Làm gọn hàm V-line perfect; V-line perfect; V-line → Tạo hình hàm, cằm V-line
- Tiêm chất làm đầy (Filler): Hyaluronic Acid (HA), HA Collagen, Restylane, Skinboosters, PROFHILO, Prolift, HA Prolift, JUVELOOK, Karisma, trẻ hóa với filler, Xóa nhăn Full Face, Tiêm hạ gò má, lấp đầy thái dương, tạo bọng mắt cười, bảo dưỡng collagen → Tiêm chất làm đầy / Tiêm HA / Tiêm trẻ hóa da
- Tiêm chất làm đầy (Filler): PLLA, PDLLA, Canxi HA (CaHA), Radiesse, PCL, mô gốc PCL PLUS, BioStimulator → Tiêm chất làm đầy kích thích sinh học
- Tiêm chất làm đầy (Filler): Botulinum Toxin → Tiêm Botulinum Toxin
- Tiêm chất làm đầy (Filler): PDRN, Rejuran, DOT PN Polynucleotide → Tiêm trẻ hóa da PDRN/Polynucleotide
- Tiêm silicon → Tiêm silicon lỏng (Thường bị cấm và nguy hiểm)
- Mesotherapy, Mesogun, MESO MELASMA, Meso Exosome, Meso Extra → Liệu pháp Mesotherapy
- Tiêm BAP, Bio Aesthetic Points → Tiêm điểm sinh học
- Nâng cơ mắt phượng; PHOENIX EYE LIFT → Tạo hình/Nâng cung mày, đuôi mắt
- Maxcollagen diamond; siêu vi điểm; Diamond Stroma → Tiêm chất làm đầy/kích thích collagen
- Laser CO2 Fractional, Laser Pico; Laser ruby; laser alexandrite; laser YAG, Duet laser, Derma V, Diode Laser, BeS Melas, laser CO2 → Điều trị da bằng Laser
- Công nghệ VIPL, IPL → Điều trị bằng ánh sáng xung cường độ cao
- shr nano light; ipl, hủy lông sinh học nano cool; triệt lạnh nano cool → Triệt lông bằng ánh sáng IPL / SHR
- Biological light technology → Liệu pháp ánh sáng sinh học
- Ánh sáng xanh; smart lux → Trị mụn bằng liệu pháp ánh sáng xanh
- Bớt OTA, Hori → Điều trị bớt sắc tố
- Hifu, LinearZ, Therapy 9D → Nâng cơ, trẻ hóa da bằng sóng siêu âm hội tụ cường độ cao (HIFU)
- Thermage FLX; RF; sóng vô tuyến derma, EXILIS; Exilis Gold Plus → Nâng cơ, trẻ hóa da bằng sóng tần số vô tuyến
- Peel GLOW SKIN, peel da, AHA; TCA; peel da hóa học → Tái tạo da bằng hóa chất
- Dermaroller → Lăn kim (Microneedling / Dermarolling)
- PRP; Platelet-Rich Plasma; huyết tương giàu tiểu cầu; cấy máu tự thân → Liệu pháp huyết tương giàu tiểu cầu (PRP)
- Nano cell; Sodium DNA, Exosome; acid admin → Liệu pháp Exosome / Tiêm PDRN
- Căng da mặt SMAS, superficial musculo aponeurotic system → Phẫu thuật căng da mặt
- Căng chỉ collagen → Căng da bằng chỉ sinh học
- Hút mỡ Ultra Z → Hút mỡ bằng sóng siêu âm
- Hút tuyến mồ hôi → Điều trị tăng tiết mồ hôi nách
- Thanh lý trái cây, treo sa trễ → Phẫu thuật treo ngực sa trễ
- Motiva Ergonomix, đặt túi → Phẫu thuật nâng ngực đặt túi
- Double Incision → Phẫu thuật tạo hình ngực theo kỹ thuật đường rạch kép
- Nâng cấp 3 vòng, thẩm mỹ 3 vòng → Phẫu thuật tạo hình cơ thể
- Thẩm mỹ "cô bé"; siêu công nghệ recover vaginal không xâm lấn → Thẩm mỹ vùng kín
- Tắm trắng Nano Bông Tuyết; tắm trắng nano white; phun trắng nano → Làm trắng da toàn thân
- Luxury Young Body → Tắm trắng / Dưỡng trắng da toàn thân
- Cryogen → Hệ thống làm lạnh Cryogen
- OxyJet → Phun oxy tươi / Jet Peel
- Cure Jet → Tiêm/Phun dưỡng chất không kim
- Carboxy CO2 therapy → Liệu pháp Carboxy (Carboxytherapy)
- Veneer, veneer IPS e.max; IPS e.max press multi; Ivoclar Vivadent; emax press → Mặt dán sứ Veneer
- Bọc sứ; răng sứ thẩm mỹ → Bọc răng sứ
- Trồng răng implant → Cấy ghép Implant nha khoa
- BioHPP → Vật liệu BioHPP (Sử dụng trong phục hình nha khoa)
- Trị răng hô, lệch lạc, chen chúc; CẢI THIỆN RĂNG CHEN CHÚC, Invisalign, Niềng răng với mắc cài, minivis; chỉnh nha → Chỉnh nha / Niềng răng
- Điều trị cười hở lợi; đánh lún chữa cười hở lợi → Điều trị cười hở lợi
- PHỤC HỒI RĂNG SÂU → Trám răng / Hàn răng
- Điêu khắc siêu sợi, điêu khắc chân mày siêu sợi, hairstroke, HAIRSTROKE EYEBROWS, microblading → Điêu khắc chân mày / Phun xăm hiệu ứng sợi
- Keo dán sinh học bio-stick → Keo dán sinh học
- Nặn mụn → Lấy nhân mụn y khoa
- 5D Skin Rejuvenation → Liệu trình trẻ hóa da đa công nghệ
- Sạch bóng nếp nhăn → Xóa nếp nhăn
- Thanh lý trái cây → Phẫu thuật loại bỏ ngực"""

# Stage 1 — Prior.
_PRIOR_PROMPT = """\
You are a compliance classifier. You output ONLY a single digit. Nothing else.

A healthcare facility in Vietnam advertises these techniques:
[Techniques]: {text_claim}

Classification rules (apply in priority order):
  - Any technique is Non-Compliant  --> output 0
  - Any technique is Partly-Compliant --> output 1
  - All techniques are Compliant    --> output 2

Output exactly one digit (0, 1, or 2). Do not output any text, punctuation, explanation, or whitespace before or after the digit.
"""

# Stage 2 — Standard expansion.
_STANDARD_EXPANSION_PROMPT = (
    """\
You are an expert in healthcare and medical regulations, specializing in verifying whether techniques or services advertised by a business comply with its registered authorized techniques or services.

You will be provided with:
- **[Allowed_Techniques]** — the registered business types of the establishment
- **[Techniques]** — techniques or services found in the advertisement
- **[Glossary]** — a reference table of aesthetic and beauty terminology to aid your assessment. This is a reference tool only — do not evaluate against it

---

## Instructions

1. For each technique in **[Techniques]**, search within **[Allowed_Techniques]** and determine whether any entry is aesthetically related to it.
   - The required level of relevance is **moderate**.
   - Focus on aesthetic relatedness, not exact wording.

2. For each matched pair, assign a **compliance level** reflecting how compliant the advertised technique is with the allowed technique.
   - The `compliance_level` must be one of: **`Non-Compliant`**, **`Partly-Compliant`**, or **`Compliant`**.
   - Use the following guidance:
     - `Compliant` — the advertised technique is clearly covered by or closely related to an allowed technique. **Also `Compliant` if the allowed technique has a broader scope that encompasses the advertised technique** (e.g., an allowed technique covers a general category that includes the advertised specific technique).
     - `Partly-Compliant` — the advertised technique is loosely or partially related to an allowed technique, but with notable differences.
     - `Non-Compliant` — no allowed technique is related to the advertised technique.
   - The `compliance_level` applies **only** to the relationship between **[Allowed_Techniques]** and **[Techniques]**.
   - **[Glossary]** is a reference tool only — do not evaluate against it.

[Allowed_Techniques]: {retrieved_regulatory_context}

[Techniques]: {text_claim}

[Glossary]:
"""
    + _GLOSSARY
    + """

---

## Output Format

Return a concise reasoning paragraph analyzing all the techniques.
"""
)

# Stage 2 — CoT-SC expansion.
_COTSC_EXPANSION_PROMPT = (
    """\
You are an expert in healthcare and medical regulations, specializing in verifying whether techniques or services advertised by a business comply with its registered authorized techniques or services.

You will be provided with:
- **[Allowed_Techniques]** — the registered business types of the establishment
- **[Techniques]** — techniques or services found in the advertisement
- **[Glossary]** — a reference table of aesthetic and beauty terminology to aid your assessment. This is a reference tool only — do not evaluate against it

---
**Instructions**

1. For each technique in **[Techniques]**, search within **[Allowed_Techniques]** and determine whether any entry is aesthetically related to it.
   - The required level of relevance is **moderate**.
   - Focus on aesthetic relatedness, not exact wording.

2. For each matched pair, assign a **compliance level** reflecting how compliant the advertised technique is with the allowed technique.
   - The `compliance_level` must be one of: **`Non-Compliant`**, **`Partly-Compliant`**, or **`Compliant`**.
   - Use the following guidance:
     - `Compliant` — the advertised technique is clearly covered by or closely related to an allowed technique. **Also `Compliant` if the allowed technique has a broader scope that encompasses the advertised technique** (e.g., an allowed technique covers a general category that includes the advertised specific technique).
     - `Partly-Compliant` — the advertised technique is loosely or partially related to an allowed technique, but with notable differences.
     - `Non-Compliant` — no allowed technique is related to the advertised technique.
   - The `compliance_level` applies **only** to the relationship between **[Allowed_Techniques]** and **[Techniques]**.
   - **[Glossary]** is a reference tool only — do not evaluate against it.

**Reasoning Framework:**

1. Identify the matched **[Allowed_Techniques]** entry for the advertised technique; consult **[Glossary]** only to clarify unfamiliar terminology, not as a compliance benchmark.
2. Explicitly state the scope direction between the two.
3. Map the scope direction to the correct compliance level.
- Reason through **ALL** techniques; partial analysis is invalid.
- Each step MUST explicitly cite which **[Allowed_Techniques]** entry applies.
- Do NOT produce a final compliance verdict — a head judge downstream will independently render the verdict after reading your analysis.

[Allowed_Techniques]: {retrieved_regulatory_context}

[Techniques]: {text_claim}

[Glossary]:
"""
    + _GLOSSARY
    + """

Task: Produce a rigorous, step-by-step analytical reasoning chain that maps each technique in [Techniques] to the relevant entry in [Allowed_Techniques].

Rules:
- Reason through **ALL** techniques; partial analysis is invalid.
- Each step MUST explicitly cite which **[Allowed_Techniques]** entry applies.
- You MUST explicitly state the scope direction before assigning a compliance level.
- Do NOT produce a final compliance verdict — a head judge downstream will independently render the verdict after reading your analysis.
- **[Glossary]** may be referenced to clarify terminology but must not be treated as a compliance standard.
- Do NOT produce a final compliance verdict — a head judge downstream will independently render the verdict after reading your analysis.
"""
)

# Stage 2 — ToT thought generation (the JSON response schema is appended verbatim).
_TOT_THOUGHT_GEN_PROMPT = (
    """\
You are an expert in healthcare and medical regulations, specializing in verifying whether techniques or services advertised by a business comply with its registered authorized techniques or services.

You will be provided with:
- **[Allowed_Techniques]** — the registered business types of the establishment
- **[Techniques]** — techniques or services found in the advertisement
- **[Glossary]** — a reference table of aesthetic and beauty terminology to aid your assessment. This is a reference tool only — do not evaluate against it

---

## Instructions

1. For each technique in **[Techniques]**, search within **[Allowed_Techniques]** and determine whether any entry is aesthetically related to it.
   - The required level of relevance is **moderate**.
   - Focus on aesthetic relatedness, not exact wording.

2. For each matched pair, assign a **compliance level** reflecting how compliant the advertised technique is with the allowed technique.
   - The `compliance_level` must be one of: **`Non-Compliant`**, **`Partly-Compliant`**, or **`Compliant`**.
   - Use the following guidance:
     - `Compliant` — the advertised technique is clearly covered by or closely related to an allowed technique. **Also `Compliant` if the allowed technique has a broader scope that encompasses the advertised technique** (e.g., an allowed technique covers a general category that includes the advertised specific technique).
     - `Partly-Compliant` — the advertised technique is loosely or partially related to an allowed technique, but with notable differences.
     - `Non-Compliant` — no allowed technique is related to the advertised technique.
   - The `compliance_level` applies **only** to the relationship between **[Allowed_Techniques]** and **[Techniques]**.
   - **[Glossary]** is a reference tool only — do not evaluate against it.

[Allowed_Techniques]: {retrieved_regulatory_context}

[Techniques]: {text_claim}

[Glossary]:
"""
    + _GLOSSARY
    + """

Reasoning so far:
{state}

Your ONLY job is to analytically map the techniques to the facility's identity and assess
their invasive/therapeutic nature. Do NOT draw the final compliance conclusion — a head
judge downstream will independently read your investigation and produce the verdict.

Task: Reason step-by-step about whether the techniques in [Techniques] are covered by, or aesthetically related to, the [Allowed_Techniques].
Set terminate=true once your investigation is comprehensive enough for a head judge to render a verdict.

Rules:
- Each thought MUST explicitly map the advertised technique to an Allowed_Technique and assess their aesthetic relatedness.
- Use ONLY the [Allowed_Techniques] and [Glossary] as evidence.

Continue the analysis. If all techniques have been analyzed and a conclusion is clear, set terminate=true and provide a final clean and concise reasoning chain that maps each technique in [Techniques] to the relevant allowed technique.


"""
    + THOUGHT_GEN_SCHEMA
)

# Stage 2 — ToT branch evaluation (the JSON response schema is appended verbatim).
_TOT_EVAL_PROMPT = (
    """\
You are an expert in healthcare and medical regulations, specializing in verifying whether techniques or services advertised by a business comply with its registered authorized techniques or services.

You will be provided with:
- **[Allowed_Techniques]** — the registered business types or expected techniques of the establishment
- **[Techniques]** — techniques or services found in the advertisement
- **[Glossary]** — a reference table of aesthetic and beauty terminology to aid your assessment. This is a reference tool only — do not evaluate against it

---

## Instructions

1. For each technique in **[Techniques]**, search within **[Allowed_Techniques]** and determine whether any entry is aesthetically related to it.
   - The required level of relevance is **moderate**.
   - Focus on aesthetic relatedness, not exact wording.

2. For each matched pair, assign a **compliance level** reflecting how compliant the advertised technique is with the allowed technique.
   - The `compliance_level` must be one of: **`Non-Compliant`**, **`Partly-Compliant`**, or **`Compliant`**.
   - Use the following guidance:
     - `Compliant` — the advertised technique is clearly covered by or closely related to an allowed technique. **Also `Compliant` if the allowed technique has a broader scope that encompasses the advertised technique** (e.g., an allowed technique covers a general category that includes the advertised specific technique).
     - `Partly-Compliant` — the advertised technique is loosely or partially related to an allowed technique, but with notable differences.
     - `Non-Compliant` — no allowed technique is related to the advertised technique.
   - The `compliance_level` applies **only** to the relationship between **[Allowed_Techniques]** and **[Techniques]**.
   - **[Glossary]** is a reference tool only — do not evaluate against it.

[Allowed_Techniques]: {retrieved_regulatory_context}

[Techniques]: {text_claim}

[Glossary]:
"""
    + _GLOSSARY
    + """

Task: You are evaluating an intermediate Tree-of-Thought branch for factual quality.

Current reasoning branch:
\"\"\"
{state}
\"\"\"

Instructions:
- Rate the factual quality of the reasoning path, not the final decision.
- Score HIGH: Path correctly follows the instructions to analyze the aesthetic relatedness between the advertised techniques and allowed techniques for ALL techniques.
- Score LOW: Path jumps to conclusions, relies on outside knowledge, or ignores techniques.
- The reasoning must be short and concise.

"""
    + STATE_EVAL_SCHEMA
)

# Stage 3 — Scoring.
_SCORING_PROMPT = (
    """\
You are a compliance classifier. You output ONLY a single digit. Nothing else.

Context:
  [Allowed_Services]: {retrieved_regulatory_context}
  [Techniques]: {text_claim}

[Glossary]:
"""
    + _GLOSSARY
    + """

Reasoning:
\"\"\"
{state}
\"\"\"

Classification rules (apply strictly in priority order, based ONLY on individual technique compliance_level values):
  - If ANY technique has compliance_level "Non-Compliant"  --> output 0
  - If ANY technique has compliance_level "Partly-Compliant" --> output 1
  - If ALL techniques have compliance_level "Compliant"    --> output 2

The first token/word is exactly one digit (0, 1, or 2). Do not output any text, punctuation, explanation, or whitespace before or after the digit.

Final Answer (0, 1, or 2):"""
)

# Stage 5 — Explanation.
_EXPLANATION_PROMPT = """\
You are a fact-checking auditor.

Evidence: {retrieved_regulatory_context}
Claim: {text_claim}
Verdict: {verdict}

Supporting reasoning:
{paths}

Write 2-3 sentences explaining why the claim is {verdict} in Vietnamese. Do not mention variable names."""


HEALTHCARE_PACKS: dict[ExpansionName, PromptPack] = {
    "standard": PromptPack(
        domain=RegulatoryDomain.HEALTHCARE,
        expansion="standard",
        prior=_PRIOR_PROMPT,
        expansion_prompt=_STANDARD_EXPANSION_PROMPT,
        scoring=_SCORING_PROMPT,
        explanation=_EXPLANATION_PROMPT,
        n_paths=3,
        expansion_max_tokens=512,
        expansion_temperature=0.7,
    ),
    "cot-sc": PromptPack(
        domain=RegulatoryDomain.HEALTHCARE,
        expansion="cot-sc",
        prior=_PRIOR_PROMPT,
        expansion_prompt=_COTSC_EXPANSION_PROMPT,
        scoring=_SCORING_PROMPT,
        explanation=_EXPLANATION_PROMPT,
        n_paths=3,
        expansion_max_tokens=2000,
        expansion_temperature=0.7,
    ),
    "tot": PromptPack(
        domain=RegulatoryDomain.HEALTHCARE,
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
