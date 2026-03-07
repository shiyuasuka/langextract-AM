# -*- coding: utf-8 -*-
"""
高熵合金论文抽取 —— Pydantic 数据模型 + Prompt 定义 + 转换工具。

核心设计：
  - Pydantic 模型的 Field(description=...) 同时充当「给人看的文档」和「给 LLM 看的指令」；
  - analysis_thought 字段专为 ERNIE 5.0 Thinking 设计：
    在 JSON 内部留一个"思维链缓存"位，防止 CoT 吐在 JSON 结构外导致解析失败；
  - LangExtract 使用扁平 Extraction → 本模块负责聚合成嵌套 MaterialEntity → 再转目标模板。
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# 0. 材料角色枚举（去噪过滤用）
# ============================================================

class MaterialRole(str, Enum):
  """材料在文中的角色，用于过滤仅保留本文研究材料。"""
  TARGET = "Target"      # 本文作者亲自制备和研究的主要材料
  REFERENCE = "Reference"  # 仅用于对比的参考文献材料
  OTHER = "Other"        # 其他不明确


# ============================================================
# 1. 基础组件
# ============================================================

class Element(BaseModel):
  """单个合金元素及其含量。"""
  symbol: str = Field(
      ...,
      description="元素符号 (e.g., 'Ni', 'Al'). 必须是标准化学符号。",
  )
  value: float = Field(
      ...,
      description="数值。如果是范围值(20-30)，请取平均值。",
  )
  unit: str = Field(
      "at.%",
      description="单位 (e.g., 'at.%', 'wt.%').",
  )
  is_balance: bool = Field(
      False,
      description="如果文中提到该元素是 'balance' 或 'rem.', 设为 True。",
  )


class Property(BaseModel):
  """单条力学性能测量。"""
  property_type: str = Field(
      ...,
      description=(
          "性能类别 (e.g., 'Yield_Strength', 'UTS', "
          "'Elongation_Total', 'Elongation_Uniform', 'Hardness')."
      ),
  )
  value: float = Field(..., description="具体数值。")
  unit: str = Field(..., description="单位 (e.g., 'MPa', '%', 'HV').")
  test_temperature: Optional[str] = Field(
      None,
      description=(
          "测试温度 (e.g., 'Room Temperature', '298K', '600C'). "
          "如未提及则留空。"
      ),
  )


class Processing(BaseModel):
  """制备工艺信息。"""
  method: str = Field(
      ...,
      description="主要制备方法 (e.g., 'Arc Melting', 'DED', 'SLM').",
  )
  heat_treatment: Optional[str] = Field(
      None,
      description="热处理条件描述 (e.g., 'Annealed at 1100C for 2h').",
  )
  details: Optional[str] = Field(
      None,
      description="其他关键工艺参数 (e.g., power, speed, layer thickness)，供 Process_Text_For_AI。",
  )


# ============================================================
# 2. 顶层聚合对象
# ============================================================

class MaterialEntity(BaseModel):
  """一种材料的完整记录。"""
  material_name: str = Field(
      ...,
      description="文中使用的材料标识符 (e.g., 'RHEA-1', 'Sample A', 'T42').",
  )
  formula: str = Field(
      ...,
      description="完整的化学式 (e.g., 'Ti42Hf21Nb21V16').",
  )
  composition: list[Element] = Field(
      default_factory=list,
      description="该材料的详细成分列表。",
  )
  process: Processing = Field(
      default_factory=lambda: Processing(method="Unknown"),
      description="该材料的制备与处理工艺。",
  )
  properties: list[Property] = Field(
      default_factory=list,
      description="该材料对应的所有力学性能数据。",
  )
  microstructure: Optional[str] = Field(
      None,
      description="微观组织描述 (e.g., 晶粒、析出相)，对应 Microstructure_Text_For_AI；暂无则留空。",
  )
  main_phase: Optional[str] = Field(
      None,
      description="主相 (e.g., 'BCC', 'FCC', 'FCC + L12').",
  )
  grain_size_um: Optional[float] = Field(
      None,
      description="平均晶粒尺寸 (μm)。",
  )
  has_precipitates: bool = Field(
      False,
      description="是否存在析出相。",
  )
  key_params_json: Optional[str] = Field(
      None,
      description="结构化工艺参数 JSON 字符串。",
  )
  process_category: str = Field(
      default="Unknown",
      description="标准化工艺分类 (AM_DED, AM_LPBF, Arc_Melting, …).",
  )
  role: str = Field(
      default=MaterialRole.OTHER.value,
      description=(
          "判断该材料在文中的角色。'Target' 表示本文作者亲自制备和研究的主要材料；"
          "'Reference' 表示仅用于对比的参考文献材料；'Other' 表示其他。"
      ),
  )


# ============================================================
# 3. 证据（溯源）记录
# ============================================================

class EvidenceSpan(BaseModel):
  """LangExtract 提供的原文溯源区间。"""
  extraction_class: str
  text: str
  char_start: Optional[int] = None
  char_end: Optional[int] = None
  alignment: Optional[str] = None


# ============================================================
# 4. LangExtract Prompt 描述（嵌入 Pydantic description）
# ============================================================

def build_prompt_description() -> str:
  """
  自动从 Pydantic 模型的 Field description 生成 LangExtract 用的 prompt_description，
  确保 Prompt 与 Schema 始终一致。
  """
  return """\
You are a materials science data engineering expert specializing in \
High-Entropy Alloys (HEA/MPEA) and Additive Manufacturing (AM). \
Extract ALL materials, compositions, processing methods, microstructure, \
and mechanical properties from this text into structured data.

IMPORTANT: Search the ENTIRE text — abstract, body, tables, AND figure \
captions — to capture every temperature point and every alloy variant.

═══════════════════════════════════════════════════════════════════
Extraction classes and required attributes
═══════════════════════════════════════════════════════════════════

1. "composition" — Chemical composition of each alloy.
   Attributes:
     material_id       — short identifier (e.g. T42, Mo3, SD3230, Mo-0)
     formula           — normalized formula WITHOUT subscript symbols
                         (e.g. Ti42Hf21Nb21V16, NOT Ti₄₂Hf₂₁).
     elements_json     — JSON string mapping element symbols to at.% values,
                         e.g. '{"Ti": 42, "Hf": 21, "Nb": 21, "V": 16}'.
                         If an element is 'balance'/'rem.', set value to -1.
     unit              — 'at.%' or 'wt.%'
     role              — 'Target' if the authors fabricated and studied this
                         material; 'Reference' if only cited for comparison;
                         'Other' if unclear. Must be exactly one of these.

2. "process" — Fabrication / processing method.
   Attributes:
     material_id       — same id as the related composition
     method            — standardized process category. Use these conventions:
                         • Laser cladding / LENS / DED → AM_DED
                         • SLM / LPBF / L-PBF → AM_LPBF
                         • Arc Melting → Arc_Melting
                         • Powder Metallurgy → PM
                         If post-processing exists, append suffix:
                         e.g. AM_LPBF_HeatTreated, AM_DED_Annealed
     heat_treatment    — heat treatment conditions (e.g. '1150C 2h Air Cooled')
     details           — other key process text (atmosphere, powder info, etc.)
     key_params_json   — JSON string of ALL numeric fabrication parameters, e.g.
                         '{"Laser_Power_W": 550, "Scanning_Speed_mm_s": 5,
                           "Hatch_Spacing_um": 381, "Layer_Thickness_um": 200,
                           "Preheating_Temp_C": 120}'.
                         Extract every numeric parameter explicitly stated.

3. "microstructure" — Microstructure and phase information.
   Attributes:
     material_id       — same id as the related composition
     main_phase        — primary crystal structure: 'BCC', 'FCC', 'FCC + L12',
                         'HCP', etc. If multiple phases, join with ' + '.
     grain_size_um     — average grain size in micrometers (numeric string).
                         Leave empty if not explicitly stated.
     has_precipitates  — 'true' if ANY precipitate / second phase is present
                         (e.g. NbC, sigma, mu, Laves, L12), 'false' otherwise.
                         If precipitates exist, also mention them in description.
     description       — concise microstructure summary: grain morphology,
                         texture, dislocation structures, precipitate type/size,
                         chemical fluctuations, etc.

4. "property" — Each individual mechanical property measurement.
   Attributes:
     material_id       — same id as the related composition
     property_type     — MUST be one of these standardized names:
                         Yield_Strength, Ultimate_Tensile_Strength,
                         Fracture_Strain, Elongation, Uniform_Elongation,
                         Total_Elongation,
                         Yield_Strength_Compressive, Ultimate_Strength_Compressive,
                         Elongation_Compressive,
                         Hardness, Hardness_HV, Hardness_HRC,
                         Density, Elastic_Modulus.
                         Do NOT use abbreviations (UTS → Ultimate_Tensile_Strength).
     value             — numeric value as a string (e.g. '1030')
     unit              — 'MPa', '%', 'HV', 'GPa', etc.
     test_temperature  — ALWAYS in Kelvin with unit suffix.
                         Room Temp / RT / 25°C → '298 K'
                         600°C → '873 K'  (add 273)
                         673 K stays '673 K'

═══════════════════════════════════════════════════════════════════
Strict Rules
═══════════════════════════════════════════════════════════════════
- Use EXACT text spans. Do NOT paraphrase or synthesize.
- List extractions in order of appearance. Do NOT overlap spans.
- Use the SAME material_id across all 4 extraction classes.
- Only extract data EXPLICITLY stated. Do NOT guess or calculate.
- If multiple alloys are studied (e.g. Mo0, Mo1, Mo3, Mo5),
  extract ALL of them with separate material_ids.
- ANISOTROPY: If the paper distinguishes Parallel/Z vs Perpendicular/X-Y
  directions, create SEPARATE property extractions noting direction in
  test_temperature (e.g. '298 K Horizontal', '298 K Vertical').
- MULTI-CONDITION: If a material has As-Built AND Heat-Treated states,
  create separate process extractions (method suffixed, e.g. AM_LPBF
  vs AM_LPBF_HeatTreated) with corresponding separate properties.
- For range values (20-30), use midpoint (25).
- For 'balance' elements, set value to -1.
- DEDUP: Do NOT extract the same (material, temperature, property_type,
  value) combination twice. Each unique measurement appears once.
"""


# ============================================================
# 5. 扁平 Extraction → MaterialEntity 聚合
# ============================================================

def _parse_elements_json(raw: str) -> list[Element]:
  """把 '{"Ti": 42, "Hf": 21}' 解析为 Element 列表。"""
  try:
    d = json.loads(raw)
    return [
        Element(
            symbol=str(k),
            value=float(v) if float(v) != -1 else 0,
            unit="at.%",
            is_balance=(float(v) == -1),
        )
        for k, v in d.items()
    ]
  except (json.JSONDecodeError, TypeError, ValueError):
    return []


def group_extractions_to_entities(
    extractions: list,
) -> tuple[list[MaterialEntity], list[EvidenceSpan]]:
  """
  将 LangExtract 返回的扁平 Extraction 按 material_id 聚合为 MaterialEntity。

  分组策略：
    - attributes 中有 material_id 的，按其值分组；
    - 缺少 material_id 的，归入最近一次 composition 的组。
  """
  all_evidence: list[EvidenceSpan] = []
  for ext in extractions:
    ci = ext.char_interval
    all_evidence.append(EvidenceSpan(
        extraction_class=ext.extraction_class,
        text=ext.extraction_text,
        char_start=ci.start_pos if ci else None,
        char_end=ci.end_pos if ci else None,
        alignment=ext.alignment_status.value if ext.alignment_status else None,
    ))

  groups: dict[str, dict[str, list]] = {}
  current_mid = "__default__"

  for ext in extractions:
    attrs = ext.attributes or {}
    mid = attrs.get("material_id", "")
    cls = ext.extraction_class

    if cls == "composition":
      mid = mid or attrs.get("formula", ext.extraction_text[:40])
      current_mid = mid
    if not mid:
      mid = current_mid

    if mid not in groups:
      groups[mid] = {"compositions": [], "processes": [], "properties": [], "microstructures": []}

    if cls == "composition":
      groups[mid]["compositions"].append(attrs)
    elif cls == "process":
      groups[mid]["processes"].append(attrs)
    elif cls == "property":
      groups[mid]["properties"].append(attrs)
    elif cls == "microstructure":
      groups[mid]["microstructures"].append(attrs)

  entities: list[MaterialEntity] = []
  for mid, g in groups.items():
    # --- composition (含 role) ---
    formula = ""
    elements: list[Element] = []
    unit = "at.%"
    role_val = MaterialRole.OTHER.value
    for c in g["compositions"]:
      raw_f = c.get("formula", "")
      if raw_f:
        formula = _normalize_formula(raw_f)
      unit = c.get("unit", unit)
      r = c.get("role", "").strip()
      if r in (MaterialRole.TARGET.value, MaterialRole.REFERENCE.value, MaterialRole.OTHER.value):
        role_val = r
      elems = _parse_elements_json(c.get("elements_json", "{}"))
      if elems:
        elements = elems
        for e in elements:
          e.unit = unit

    # --- process ---
    method = ""
    heat_treatment = None
    details_parts = []
    for p in g["processes"]:
      method = p.get("method", method)
      ht = p.get("heat_treatment")
      if ht:
        heat_treatment = ht
      det = p.get("details")
      if det:
        details_parts.append(det)

    details_joined = " ".join(details_parts).strip() if details_parts else None

    key_params_merged: dict = {}
    for p in g["processes"]:
      kp_raw = p.get("key_params_json", "")
      if kp_raw:
        try:
          kp_dict = json.loads(kp_raw) if isinstance(kp_raw, str) else kp_raw
          if isinstance(kp_dict, dict):
            key_params_merged.update(kp_dict)
        except (json.JSONDecodeError, TypeError):
          pass

    proc = Processing(
        method=method or "Unknown",
        heat_treatment=heat_treatment,
        details=details_joined,
    )
    proc_category = _standardize_process_category(method, heat_treatment)

    # --- microstructure ---
    main_phase = ""
    grain_size_um = None
    has_precipitates = False
    micro_descriptions: list[str] = []
    for ms in g.get("microstructures", []):
      phase = ms.get("main_phase", "").strip()
      if phase:
        main_phase = phase
      gs = ms.get("grain_size_um", "")
      if gs:
        try:
          grain_size_um = float(gs)
        except (ValueError, TypeError):
          pass
      hp = ms.get("has_precipitates", "").strip().lower()
      if hp in ("true", "yes", "1"):
        has_precipitates = True
      desc = ms.get("description", "").strip()
      if desc:
        micro_descriptions.append(desc)

    microstructure_text = " ".join(micro_descriptions).strip() or None

    # --- properties (with dedup and standardization) ---
    props: list[Property] = []
    seen_props: set[tuple] = set()
    for pr in g["properties"]:
      try:
        val = float(pr.get("value", ""))
      except (ValueError, TypeError):
        continue
      ptype = _standardize_property_type(
          pr.get("property_type", pr.get("name", "Unknown"))
      )
      temp_str = pr.get("test_temperature", pr.get("condition"))
      temp_k = _parse_temp_to_k(temp_str)
      dedup_key = (ptype, val, pr.get("unit", ""), temp_k)
      if dedup_key in seen_props:
        continue
      seen_props.add(dedup_key)
      props.append(Property(
          property_type=ptype,
          value=val,
          unit=pr.get("unit", ""),
          test_temperature=temp_str,
      ))

    entity = MaterialEntity(
        material_name=mid,
        formula=_normalize_formula(formula) if formula else mid,
        composition=elements,
        process=proc,
        properties=props,
        microstructure=microstructure_text,
        main_phase=main_phase or None,
        grain_size_um=grain_size_um,
        has_precipitates=has_precipitates,
        key_params_json=json.dumps(key_params_merged, ensure_ascii=False) if key_params_merged else None,
        process_category=proc_category,
        role=role_val,
    )
    entities.append(entity)

  return entities, all_evidence


# ============================================================
# 6. MaterialEntity → 用户目标 JSON 模板
# ============================================================

# ---- 工艺分类标准化 ----

_PROCESS_CATEGORY_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:laser\s*cladding|LENS|DED|L-?DED|DMD|WAAM)", re.I), "AM_DED"),
    (re.compile(r"(?:SLM|L-?PBF|LPBF|DMLS)", re.I),                   "AM_LPBF"),
    (re.compile(r"(?:EBM|E-?PBF)", re.I),                              "AM_EBM"),
    (re.compile(r"(?:arc\s*melting|VAM)", re.I),                        "Arc_Melting"),
    (re.compile(r"(?:cast(?:ing)?|induction\s*melting)", re.I),         "Casting"),
    (re.compile(r"(?:powder\s*metallurgy|HIP|SPS|MA)", re.I),          "PM"),
    (re.compile(r"(?:cold\s*roll|hot\s*roll|rolling)", re.I),          "Rolling"),
    (re.compile(r"(?:forging|forged)", re.I),                          "Forging"),
]

_HT_PATTERN = re.compile(
    r"(?:anneal|heat.?treat|homogeni|solution|aging|temper|quench)", re.I
)


def _standardize_process_category(method: str, heat_treatment: str | None) -> str:
    """将原始工艺描述映射为标准化 Process_Category。"""
    combined = method
    if heat_treatment:
        combined = f"{method} {heat_treatment}"

    base = "Unknown"
    for pat, cat in _PROCESS_CATEGORY_MAP:
        if pat.search(combined):
            base = cat
            break

    if _HT_PATTERN.search(heat_treatment or ""):
        return f"{base}_HeatTreated"
    return base


# ---- 化学式下标符号去除 ----

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _normalize_formula(raw: str) -> str:
    """去除 Unicode 下标符号并清理空白，如 Ti₄₂Hf₂₁ → Ti42Hf21。"""
    return raw.translate(_SUBSCRIPT_MAP).replace("$", "").strip()


# ---- 性能类型标准化 ----

_PROPERTY_TYPE_MAP = {
    "uts": "Ultimate_Tensile_Strength",
    "ultimate tensile strength": "Ultimate_Tensile_Strength",
    "tensile strength": "Ultimate_Tensile_Strength",
    "yield strength": "Yield_Strength",
    "yield_strength": "Yield_Strength",
    "compressive yield strength": "Yield_Strength_Compressive",
    "yield_strength_compressive": "Yield_Strength_Compressive",
    "compressive strength": "Ultimate_Strength_Compressive",
    "ultimate_strength_compressive": "Ultimate_Strength_Compressive",
    "ultimate compressive strength": "Ultimate_Strength_Compressive",
    "elongation": "Elongation",
    "elongation_total": "Total_Elongation",
    "total elongation": "Total_Elongation",
    "total_elongation": "Total_Elongation",
    "uniform elongation": "Uniform_Elongation",
    "uniform_elongation": "Uniform_Elongation",
    "elongation_uniform": "Uniform_Elongation",
    "fracture strain": "Fracture_Strain",
    "fracture_strain": "Fracture_Strain",
    "elongation_compressive": "Elongation_Compressive",
    "compressive elongation": "Elongation_Compressive",
    "compressive strain": "Elongation_Compressive",
    "hardness": "Hardness",
    "hardness_hv": "Hardness_HV",
    "hardness_hrc": "Hardness_HRC",
    "elastic modulus": "Elastic_Modulus",
    "young's modulus": "Elastic_Modulus",
    "density": "Density",
}


def _standardize_property_type(raw: str) -> str:
    """将常见的 property_type 变体映射为标准名称。"""
    key = raw.strip().lower()
    return _PROPERTY_TYPE_MAP.get(key, raw.strip())


def _parse_temp_to_k(temp_str: str | None) -> float:
  """温度字符串转开尔文: RT/room->298, 1000C->1273.15, 298K->298。"""
  if not temp_str:
    return 298.0
  t = temp_str.lower().strip()
  if "rt" in t or "room" in t:
    return 298.0
  nums = re.findall(r"[\d.]+", temp_str)
  if not nums:
    return 298.0
  val = float(nums[0])
  if "k" in t and "c" not in t:
    return val
  return val + 273.15  # 默认按摄氏度


def _extract_direction(temp_str: str | None) -> str | None:
  """从温度字符串中提取方向信息（如 Horizontal / Vertical / Z / X-Y）。"""
  if not temp_str:
    return None
  t = temp_str.lower()
  for tag, label in [
      ("horizontal", "Horizontal"), ("vertical", "Vertical"),
      ("parallel", "Parallel_Z"), ("perpendicular", "Perpendicular_XY"),
      (" z ", "Parallel_Z"), (" x-y", "Perpendicular_XY"),
      (" xy", "Perpendicular_XY"),
  ]:
    if tag in t:
      return label
  return None


def entity_to_target_json(
    entity: MaterialEntity,
    source_pdf: str,
    evidence: list[EvidenceSpan] | None = None,
) -> dict[str, Any]:
  """
  将 MaterialEntity 转为甲方严格 JSON 模板。
  一级 Key: Composition_Info, Process_Info, Properties_Info。
  Composition_JSON / Key_Params_JSON 使用 json.dumps(..., ensure_ascii=False) 生成转义字符串。
  """
  safe_name = re.sub(r"[^a-zA-Z0-9]", "", entity.material_name)[:15] or "Unknown"
  mat_id = f"M_{safe_name}"
  ht_tag = "HT" if _HT_PATTERN.search(entity.process.heat_treatment or "") else "AB"
  sample_id = f"S_{safe_name}_{ht_tag}"

  comp_dict = {
      e.symbol: (-1 if e.is_balance else e.value)
      for e in entity.composition
  }

  composition_info = {
      "Mat_ID": mat_id,
      "Alloy_Name_Raw": entity.material_name,
      "Formula_Normalized": _normalize_formula(entity.formula),
      "Composition_JSON": json.dumps(comp_dict, ensure_ascii=False),
      "Source_DOI": source_pdf,
  }

  process_text_parts = []
  if entity.process.method and entity.process.method != "Unknown":
    process_text_parts.append(entity.process.method)
  if entity.process.heat_treatment:
    process_text_parts.append(entity.process.heat_treatment)
  if entity.process.details:
    process_text_parts.append(entity.process.details)
  if entity.has_precipitates and entity.microstructure:
    process_text_parts.append(f"Precipitates observed: {entity.microstructure}")
  process_text = ". ".join(process_text_parts) if process_text_parts else ""

  process_info = {
      "Sample_ID": sample_id,
      "Mat_ID": mat_id,
      "Process_Category": entity.process_category,
      "Process_Text_For_AI": process_text,
      "Key_Params_JSON": entity.key_params_json or "{}",
      "Main_Phase": entity.main_phase or "",
      "Microstructure_Text_For_AI": entity.microstructure or "",
      "Has_Precipitates": entity.has_precipitates,
      "Grain_Size_avg_um": entity.grain_size_um,
  }

  props: list[dict[str, Any]] = []
  for i, p in enumerate(entity.properties, 1):
    direction = _extract_direction(p.test_temperature)
    entry: dict[str, Any] = {
        "Test_ID": f"T_{safe_name}_{i:02d}",
        "Sample_ID": sample_id,
        "Test_Temperature_K": _parse_temp_to_k(p.test_temperature),
        "Property_Type": p.property_type,
        "Property_Value": p.value,
        "Property_Unit": p.unit,
    }
    if direction:
      entry["Test_Direction"] = direction
    props.append(entry)

  result: dict[str, Any] = {
      "_source_pdf": source_pdf,
      "role": getattr(entity, "role", MaterialRole.OTHER.value),
      "Composition_Info": composition_info,
      "Process_Info": process_info,
      "Properties_Info": props,
  }
  if evidence:
    result["_evidence"] = [e.model_dump() for e in evidence]

  return result


def material_entity_to_target_json(entity: MaterialEntity, pdf_name: str) -> dict[str, Any]:
  """兼容接口：仅根据实体与 PDF 名称返回甲方 JSON 结构。"""
  return entity_to_target_json(entity=entity, source_pdf=pdf_name, evidence=None)
