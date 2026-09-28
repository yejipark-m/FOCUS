"""
MMLongBench evaluation.
Loads data from /root/share/MMLongBench/mmlb_data and mmlb_image directly —
no HuggingFace datasets, no imports from the MMLongBench repo.
"""
import re
import json
import os
import string
import random
import math
import argparse
from collections import defaultdict

import yaml
from PIL import Image
from transformers import set_seed
from tqdm import tqdm

from model.model_handler import ModelHandler
from utils.cd_sample import evolve_cd_sampling
from utils.add_noise import NoiseProcessor


MMLB_DATA_ROOT  = "/root/share/MMLongBench/mmlb_data"
MMLB_IMAGE_ROOT = "/root/share/MMLongBench/mmlb_image"


# ──────────────────────────────────────────────────────────────
# Metric utilities  (ported from MMLongBench/utils.py)
# ──────────────────────────────────────────────────────────────

def _normalize(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())

def _normalize_punc(s):
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s.lower()).split())

def _f1(pred, gold):
    from collections import Counter
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0, 0.0, 0.0
    p = n / len(pt)
    r = n / len(gt)
    return (2 * p * r) / (p + r), p, r

def _sub_em(pred, gold):
    return _normalize(gold) in _normalize(pred)

def _max_over_golds(fn, pred, golds):
    if isinstance(golds, str):
        golds = [golds]
    elif golds and isinstance(golds[0], list):
        golds = [g for gl in golds for g in gl]
    return max(fn(pred, g) for g in golds)

def _binary_label(text):
    text = _normalize_punc(text)
    m = re.search(r"\b(?:yes|no)\b(?!.*\b(?:yes|no)\b)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return {"yes": 1, "no": 0}[m.group(0).lower()]
    return int(text.strip()) if text.strip().isdigit() else -1

def _cnt_list(pred):
    pred = _normalize_punc(pred)
    for pat in (r'\[[\d\s,]+\]', r'\[.*?\]'):
        ms = re.findall(pat, pred, re.IGNORECASE | re.DOTALL)
        if ms:
            try:
                lst = json.loads(ms[-1])
                return [int(x) for x in lst if str(x).lstrip('-').isdigit()]
            except (json.JSONDecodeError, ValueError):
                pass
    return [int(x) for x in re.findall(r'\d+', pred)]

def _choice_letter(pred):
    pred = ' '.join(pred.split())
    m = re.search(r"answer is \(?([A-J])\)?", pred)
    if m:
        return ord(m.group(1)) - ord('A')
    m = re.search(r'[aA]nswer:\s*([A-J])', pred)
    if m:
        return ord(m.group(1)) - ord('A')
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", pred, re.DOTALL)
    if m:
        return ord(m.group(0)) - ord('A')
    return -1

def _class_num(pred):
    pred = ' '.join(pred.split())
    m = re.search(r"label:\s*(\d+)", pred, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b(?!.*\b\d+\b)", pred)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', pred)
    return int(nums[0]) if nums else -1

# ── docqa helpers ─────────────────────────────────────────────

def _clean(s):
    s = str(s).lower().strip().replace(",", "")
    for sfx in ["kg", "meters", "acres", "minutes", "miles", "mile", "feet",
                "million", "thousand", "billion", "mm", "m"]:
        s = re.sub(re.escape(sfx) + r'$', '', s).strip()
    return re.sub(r"^['\"]|['\"]$", "", s).strip().strip("$").strip("£").strip("%").strip()

def _float_eq(ref, pred, pct=False):
    from math import isclose
    targets = [ref / 100, ref, ref * 100] if pct else [ref]
    return any(isclose(t, pred, rel_tol=0.01) for t in targets)

_NEED_EM = [
    r'https://', r'.*\.(py|ipynb)$', r'^page', r'^\d+(-\d+|\s\d+)?$',
    r'(a\.m\.|p\.m\.)', r'^\d{4}[-/\s]\d{1,2}[-/\s]\d{1,2}$',
    r'^\d{1,2}[-/\s]\d{1,2}[-/\s]\d{2,4}$', r'^\d{4}[-/\s]\d{1,2}$',
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
]

def _need_em(s):
    return any(re.search(p, s) for p in _NEED_EM)

def _numbers(pred):
    nums = []
    for m, _ in re.findall(r'(-?\d+(\.\d*)?|-?\.\d+)', pred.replace(',', '')):
        try:
            n = float(m)
            if math.isfinite(n):
                nums.append(n)
        except ValueError:
            pass
    return nums

def _str_type(s):
    try:
        n = float(_clean(str(s)))
        return "Integer" if n == int(n) and "%" not in str(s) else "Float"
    except Exception:
        return "String"

def eval_docqa(gt, pred, atype):
    if atype == "Integer":
        gt_int = int(float(_clean(str(gt))))
        pred_nums = [int(n) for n in _numbers(_clean(str(pred))) if int(n) == n]
        return float(any(gt_int == p for p in pred_nums))
    if atype == "Float":
        gt_f = float(_clean(str(gt)))
        return float(any(_float_eq(gt_f, p, pct=True) for p in _numbers(_clean(str(pred)))))
    if atype in ("String", "None"):
        if _need_em(gt):
            return float(gt in pred)
        return _f1(pred, gt)[0]
    if atype == "List":
        gt_list = json.loads(gt)
        str_items = [i for i in gt_list
                     if isinstance(i, str) and _str_type(i) == "String" and not _need_em(i)]
        other = [i for i in gt_list
                 if not (isinstance(i, str) and _str_type(i) == "String" and not _need_em(i))]
        effective = ([" ".join(str_items).strip()] if str_items else []) + other
        scores = []
        for g in effective:
            t = "Integer" if isinstance(g, int) else "Float" if isinstance(g, float) else _str_type(g)
            scores.append(eval_docqa(g, pred, t))
        return sum(scores) / len(effective)
    raise KeyError(f"Unknown answer type: {atype}")

def parse_output(text, prefix="Answer:"):
    def lstrip(s, sub):
        return re.sub(f'^{re.escape(sub)}', '', s, flags=re.IGNORECASE)
    for pat in (re.compile(f"(?:{prefix})(.*)", re.IGNORECASE | re.DOTALL),
                re.compile(r"(?:^)(.*)", re.IGNORECASE | re.DOTALL)):
        m = pat.search(text)
        if m:
            return lstrip(m.group(1).strip(), prefix).strip()
    return None

def strip_thinking(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


# ──────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────

def _load_img(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

def _imgpath(rel):
    return os.path.join(MMLB_IMAGE_ROOT, rel)


# ──────────────────────────────────────────────────────────────
# Task loaders
# Each returns (examples, user_template, sys_prefix).
# Each example has: question, context, image_paths, answer, post_fn.
# ──────────────────────────────────────────────────────────────

def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def _load_json(path):
    with open(path) as f:
        return json.load(f)

def _subsample(rows, max_samples, seed, key="id"):
    if max_samples is None:
        return rows
    ids = sorted({r[key] for r in rows})
    ids = random.Random(seed).sample(ids, min(max_samples, len(ids)))
    id_set = set(ids)
    return [r for r in rows if r[key] in id_set]


# ── VRAG (infoseek / viquae) ──────────────────────────────────

_VRAG_TMPL = (
    "Use the given documents to write a concise and short answer to the question "
    "about the entity shown in the image. Write your answer in the following format:\n"
    "Answer: [answer]\n\n{context}\n\nQuestion: {question}"
)
_VRAG_SYS = "Answer:"

def _vrag_post(pred, ans):
    pred = strip_thinking(pred)
    parsed = parse_output(pred, prefix=_VRAG_SYS) or pred
    s1 = int(_max_over_golds(_sub_em, pred, ans))
    s2 = int(_max_over_golds(_sub_em, parsed, ans))
    return {"sub_em": max(s1, s2), "parsed_output": parsed}

def load_vrag(path, max_samples, seed):
    rows = _load_jsonl(path)
    rows = [r for r in rows if '"' not in r["image"]]
    rows = _subsample(rows, max_samples, seed)
    examples = [
        {"question": "<image>" + r["question"],
         "context": "\n\n".join(f"Document (Title: {c['title']}): {c['text']}" for c in r["ctxs"]),
         "image_paths": [_imgpath(r["image"])],
         "answer": r["answer"], "post_fn": _vrag_post}
        for r in rows
    ]
    return examples, _VRAG_TMPL, _VRAG_SYS


# ── Visual Haystack (vh_single / vh_multi) ────────────────────

_VH_TMPL = (
    "You are given a set of images. Please answer the question in Yes or No based on "
    "the given images. Write your answer in the following format:\n"
    "Answer: [answer]\n\n{context}\n\nQuestion: {question}"
)
_VH_SYS = "Answer:"

class _DefaultAnswerGen:
    _cycle = [0, 1, 1, 0]
    def __init__(self): self._i = 0
    def __call__(self):
        v = self._cycle[self._i]; self._i = (self._i + 1) % 4; return v

def load_visual_haystack(path, max_samples, seed):
    rows = _load_jsonl(path)
    rows = _subsample(rows, max_samples, seed)
    dag = _DefaultAnswerGen()
    def post(pred, ans, _dag=dag):
        pred = strip_thinking(pred)
        default = _dag()
        label = _binary_label(pred)
        if label == -1:
            label = default
        gt = _binary_label(str(ans))
        return {"acc": int(label == gt), "parsed_output": pred, "default_answer": default}
    examples = [
        {"question": r["question"],
         "context": "\n".join(["<image>"] * len(r["ctxs"])),
         "image_paths": [_imgpath(p) for p in r["ctxs"]],
         "answer": r["answer"], "post_fn": post}
        for r in rows
    ]
    return examples, _VH_TMPL, _VH_SYS


# ── MM-NIAH text ──────────────────────────────────────────────

_NIAH_TMPL = (
    "You are given interleaved text and images. Please answer the question based on "
    "the given text and images. Write your answer in the following format:\n"
    "Answer: [answer]\n\n{context}\n\nQuestion: {question}"
)
_NIAH_SYS = "Answer:"

def _niah_text_post(metric):
    def post(pred, ans):
        pred = strip_thinking(pred)
        parsed = parse_output(pred, prefix=_NIAH_SYS) or pred
        def _calc(p):
            if metric == "sub_em":
                return {"sub_em": int(_max_over_golds(_sub_em, p, ans))}
            return {"soft_acc": int(sum(_cnt_list(p)) == sum(ans))}
        m1, m2 = _calc(pred), _calc(parsed)
        return {k: max(m1.get(k, 0), m2.get(k, 0)) for k in m1} | {"parsed_output": parsed}
    return post

def load_niah_text(path, max_samples, seed, metric="sub_em"):
    rows = _load_jsonl(path)
    rows = _subsample(rows, max_samples, seed)
    post = _niah_text_post(metric)
    examples = [
        {"question": r["question"],
         "context": "\n\n".join(p["text"] for p in r["ctxs"]),
         "image_paths": [_imgpath(p) for p in r.get("image_list", [])],
         "answer": r["answer"], "post_fn": post}
        for r in rows
    ]
    return examples, _NIAH_TMPL, _NIAH_SYS


# ── MM-NIAH image MC (retrieval-image / reasoning-image) ──────

_NIAH_MC_TMPL = (
    "You are given interleaved text and images. Please answer the question with the "
    "option's letter (A, B, etc.) based on the given text and images. Write your answer "
    "in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
)

def _niah_mc_post(pred, ans):
    pred = strip_thinking(pred)
    parsed = parse_output(pred, prefix=_NIAH_SYS) or pred
    s1 = int(_choice_letter(pred) == ans)
    s2 = int(_choice_letter(parsed) == ans)
    return {"mc_acc": max(s1, s2), "parsed_output": parsed}

def load_niah_image_mc(path, max_samples, seed):
    rows = _load_jsonl(path)
    rows = _subsample(rows, max_samples, seed)
    examples = []
    for r in rows:
        n = len(r["choices_image"])
        question = r["question"] + "".join(f"\n{chr(65+i)}. <image>" for i in range(n))
        imgs = [_imgpath(p) for p in r["image_list"]] + [_imgpath(p) for p in r["choices_image"]]
        examples.append({"question": question,
                          "context": "\n\n".join(p["text"] for p in r["ctxs"]),
                          "image_paths": imgs, "answer": r["answer"], "post_fn": _niah_mc_post})
    return examples, _NIAH_MC_TMPL, _NIAH_SYS


# ── MM-NIAH image counting ────────────────────────────────────

def _niah_count_post(pred, ans):
    pred = strip_thinking(pred)
    parsed = parse_output(pred, prefix=_NIAH_SYS) or pred
    s1 = int(sum(_cnt_list(pred)) == sum(ans))
    s2 = int(sum(_cnt_list(parsed)) == sum(ans))
    return {"soft_acc": max(s1, s2), "parsed_output": parsed}

def load_niah_image_count(path, max_samples, seed):
    rows = _load_jsonl(path)
    rows = _subsample(rows, max_samples, seed)
    examples = []
    for r in rows:
        imgs = [_imgpath(p) for p in r["image_list"]] + [_imgpath(p) for p in r["needle_image_list"]]
        examples.append({"question": r["question"],
                          "context": "\n\n".join(p["text"] for p in r["ctxs"]),
                          "image_paths": imgs, "answer": r["answer"], "post_fn": _niah_count_post})
    return examples, _NIAH_TMPL, _NIAH_SYS


# ── ICL (cars196 / food101 / inat2021 / sun397) ───────────────

_ICL_TMPL = (
    'You need to recognize entities in images. Use the provided mapping from the image '
    'to label to assign a label to the test image. Only output "label: {{label}}" and '
    'nothing else.\n\nTraining examples:\n{context}\n\nNow classify this image: {question}'
)
_ICL_SYS = "label:"

def _icl_post(pred, ans):
    pred = strip_thinking(pred)
    parsed = parse_output(pred, prefix=_ICL_SYS) or pred
    s1 = int(_class_num(pred) == ans)
    s2 = int(_class_num(parsed) == ans)
    return {"cls_acc": max(s1, s2), "parsed_output": parsed}

def load_icl(path, max_samples, seed):
    data = _load_json(path)
    rng = random.Random(seed)
    all_tests = [{"domain": d, "example": ex}
                 for d, dd in data.items() for ex in dd["test_example"]]
    exemplar_map = {d: dd["exemplar_list"] for d, dd in data.items()}
    if max_samples:
        rng.shuffle(all_tests)
        all_tests = all_tests[:max_samples]
    examples = []
    for item in all_tests:
        domain, ex = item["domain"], item["example"]
        sampled = rng.choice(exemplar_map[domain])
        for rnd in sampled:
            rng.shuffle(rnd)
        flat = [s for rnd in sampled for s in rnd]
        ctx = "\n\n".join(f"<image>\nlabel: {s['id']}" for s in flat)
        imgs = [_imgpath(s["image"]) for s in flat] + [_imgpath(ex["image"])]
        examples.append({"question": "<image>", "context": ctx,
                          "image_paths": imgs, "answer": ex["answer"], "post_fn": _icl_post})
    return examples, _ICL_TMPL, _ICL_SYS


# ── Summarization (gov-report / lexsum) ───────────────────────

_SUMM_GOV_TMPL = (
    "You are given a government report from U.S. Government Accountability Office (GAO), "
    "and you are tasked to summarize the report. Write a concise summary (around 550 words) "
    "organized in multiple paragraphs. Where applicable, the summary should contain a short "
    "description of why GAO did this study, what GAO found, and what GAO recommends.\n\n"
    "Government Report:\n{context}\n\nNow please summarize the report."
)
_SUMM_LEX_TMPL = (
    "You are given the legal documents in a civil rights lawsuit, and you are tasked to "
    "summarize the case. Write a concise summary of one paragraph (200 to 250 words). "
    "The summary should contain a short description of the background, the parties involved, "
    "and the outcomes of the case.\n\nLegal documents:\n{context}\n\nNow please summarize the case."
)
_SUMM_SYS = "Summary:"

def _page_id(path):
    try:
        return path.split("page")[1].split(".")[0]
    except IndexError:
        return os.path.splitext(os.path.basename(path))[0]

def _summ_post(pred, ans):
    from rouge_score import rouge_scorer
    pred = strip_thinking(pred)
    parsed = parse_output(pred, prefix=_SUMM_SYS) or pred
    answers = [ans] if isinstance(ans, str) else ans
    scorer = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)
    rouges = [scorer.score(target=a, prediction=parsed) for a in answers]
    return {
        "rougeL_f1":    max(r["rougeL"].fmeasure for r in rouges),
        "rougeLsum_f1": max(r["rougeLsum"].fmeasure for r in rouges),
        "parsed_output": parsed,
    }

def _summ_ctx(pages):
    return "\n\n".join(
        f"Document {p.split('/')[-2]:.15} (page {_page_id(p)}): <image>" for p in pages
    )

def load_summ_gov(path, max_samples, seed):
    rows = _load_jsonl(path)
    if max_samples:
        rows = random.Random(seed).sample(rows, min(max_samples, len(rows)))
    examples = []
    for r in rows:
        ans = "\n\n".join(
            a["section_title"] + ":\n" + "\n".join(a["paragraphs"]) for a in r["summary"]
        )
        examples.append({"question": "", "context": _summ_ctx(r["image_list"]),
                          "image_paths": [_imgpath(p) for p in r["image_list"]],
                          "answer": ans, "post_fn": _summ_post})
    return examples, _SUMM_GOV_TMPL, _SUMM_SYS

def load_summ_lex(path, max_samples, seed):
    rows = _load_jsonl(path)
    if max_samples:
        rows = random.Random(seed).sample(rows, min(max_samples, len(rows)))
    examples = [
        {"question": "", "context": _summ_ctx(r["image_list"]),
         "image_paths": [_imgpath(p) for p in r["image_list"]],
         "answer": r["summary"], "post_fn": _summ_post}
        for r in rows
    ]
    return examples, _SUMM_LEX_TMPL, _SUMM_SYS


# ── Document QA (longdocurl / mmlongdoc / slidevqa) ──────────

_DOCQA_TMPL = (
    "You are given a document with text and images, and a question. Answer the question "
    "as concisely as you can, using a single phrase or sentence if possible. If the question "
    "cannot be answered based on the information in the article, write 'Not answerable.' "
    "Write your answer in the following format:\n"
    "Answer: [answer]\n\n{context}\n\nQuestion: {question}"
)
_DOCQA_SYS = "Answer:"

def _make_docqa_post(answer_format):
    def post(pred, ans):
        pred = strip_thinking(pred)
        parsed = parse_output(pred, prefix=_DOCQA_SYS) or pred
        return {"doc_qa": eval_docqa(ans, parsed, answer_format), "parsed_output": parsed}
    return post

def load_docqa(path, max_samples, seed):
    rows = _load_jsonl(path)
    if max_samples:
        rows = random.Random(seed).sample(rows, min(max_samples, len(rows)))
    examples = []
    for r in rows:
        pages = r["page_list"]
        ctx = "\n\n".join(f"Document {p.split('/')[-2]:.15}: <image>" for p in pages)
        question = f"Based on Document {r['doc_name']:.15}, answer the following question. " + r["question"]
        examples.append({"question": question, "context": ctx,
                          "image_paths": [_imgpath(p) for p in pages],
                          "answer": r["answer"],
                          "post_fn": _make_docqa_post(r.get("answer_format", "String"))})
    return examples, _DOCQA_TMPL, _DOCQA_SYS


# ──────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────

def load_task(dataset_name, test_file, max_samples=None, seed=42):
    path  = os.path.join(MMLB_DATA_ROOT, test_file)
    fname = os.path.basename(test_file)
    dn    = dataset_name.lower()

    if "infoseek" in dn or "viquae" in dn:
        return load_vrag(path, max_samples, seed)
    if "vh_single" in dn or "vh_multi" in dn:
        return load_visual_haystack(path, max_samples, seed)
    if "mm_niah" in dn or "text-haystack" in dn:
        if "counting-image" in fname:
            return load_niah_image_count(path, max_samples, seed)
        if "retrieval-image" in fname or "reasoning-image" in fname or "text-haystack" in fname:
            return load_niah_image_mc(path, max_samples, seed)
        metric = "soft_acc" if "counting" in fname else "sub_em"
        return load_niah_text(path, max_samples, seed, metric)
    if any(k in dn for k in ("cars196", "food101", "inat2021", "sun397")):
        return load_icl(path, max_samples, seed)
    if "gov-report" in dn or "gov_report" in dn:
        return load_summ_gov(path, max_samples, seed)
    if "lexsum" in dn:
        return load_summ_lex(path, max_samples, seed)
    if any(k in dn for k in ("longdocurl", "mmlongdoc", "slidevqa")):
        return load_docqa(path, max_samples, seed)
    raise ValueError(f"Unknown dataset: {dataset_name!r}")


# ──────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_mmlongbench(model_name, dataset_name, test_file,
                          max_samples, seed, hyperparameter, noise):
    """
    Evaluates the specified vision-language model on one MMLongBench subset.
    """
    model_handler = ModelHandler(model_name=model_name)
    if hyperparameter is not None:
        hyperparameter = float(hyperparameter)
        evolve_cd_sampling(hyperparameter)
    model_handler.model.eval()

    examples, user_template, _sys = load_task(dataset_name, test_file, max_samples, seed)
    metrics_all = defaultdict(list)

    for example in tqdm(examples, desc=f"{dataset_name}"):
        images = [_load_img(p) for p in example["image_paths"]]

        if noise is not None and images:
            pure_noise, single_noise = noise
            pure_noise_imgs = [pure_noise(img) for img in images]
            injected = [
                [img if j == i else single_noise(images[j]) for j, img in enumerate(images)]
                for i in range(len(images))
            ]
            images = [pure_noise_imgs, *injected]

        prompt = user_template.format(context=example["context"], question=example["question"])
        prediction = model_handler.generate_response(images or [], instruction=prompt).strip()

        result = example["post_fn"](prediction, example["answer"])
        for k, v in result.items():
            if k not in ("parsed_output", "default_answer"):
                metrics_all[k].append(v)

        n   = len(next(iter(metrics_all.values())))
        line = "  ".join(f"{k}={sum(v)/len(v):.4f}" for k, v in metrics_all.items())
        tqdm.write(f"  [#{n:4d}]  {line}")

    return {k: sum(v) / len(v) for k, v in metrics_all.items()}


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLMs with MMLongBench.")
    parser.add_argument("--model_name", type=str, default="qwen2.5-vl-7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap per dataset subset (None = all).")
    parser.add_argument("--config", type=str, default=None,
                        help="MMLongBench YAML config, e.g. /root/share/MMLongBench/configs/vrag_all.yaml")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names.")
    parser.add_argument("--test_files", type=str, default=None,
                        help="Comma-separated test file paths relative to mmlb_data/.")
    parser.add_argument("--hyperparameter", default=None)
    parser.add_argument(
        "--noise",
        choices=["black", "white", "gray", "random", "gaussian", "uniform",
                 "imagenet+gaussian", "imagenet+uniform", "imagenet+diffusion",
                 "shot", "impulse", "speckle"],
        default="imagenet+uniform",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--severity", type=int, choices=[1, 2, 3, 4, 5], default=3)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.config is not None:
        cfg        = yaml.safe_load(open(args.config))
        datasets   = cfg["datasets"].split(",")
        test_files = cfg["test_files"].split(",")
    elif args.datasets and args.test_files:
        datasets   = args.datasets.split(",")
        test_files = args.test_files.split(",")
    else:
        parser.error("Provide either --config or both --datasets and --test_files.")

    assert len(datasets) == len(test_files)

    noise_processor = None
    if args.hyperparameter is not None:
        noise_processor = [
            NoiseProcessor(mode=args.noise, scale=1.0,        severity=args.severity),
            NoiseProcessor(mode=args.noise, scale=args.scale, severity=args.severity),
        ]

    all_results = {}
    for dataset_name, test_file in zip(datasets, test_files):
        dataset_name = dataset_name.strip()
        test_file    = test_file.strip()
        print(f"\n{'='*60}\n{dataset_name}  |  {test_file}\n{'='*60}")
        try:
            metrics = evaluate_mmlongbench(
                model_name=args.model_name,
                dataset_name=dataset_name,
                test_file=test_file,
                max_samples=args.max_samples,
                seed=args.seed,
                hyperparameter=args.hyperparameter,
                noise=noise_processor,
            )
            all_results[f"{dataset_name}|{test_file}"] = metrics
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}")

    if len(all_results) > 1:
        print("\n" + "="*60 + "\nSummary:")
        for key, metrics in all_results.items():
            print(f"  {key}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
