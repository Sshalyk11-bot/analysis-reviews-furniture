from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, mannwhitneyu

try:
    import pymorphy3
    MORPH = pymorphy3.MorphAnalyzer()
    NORMALIZER = "pymorphy3"
except ImportError:
    MORPH = None
    from nltk.stem.snowball import RussianStemmer
    STEMMER = RussianStemmer()
    NORMALIZER = "snowball_stemmer_fallback"

INPUT_XLSX = Path(os.getenv("INPUT_XLSX", "reviews_yandex_1.xlsx"))
OUTPUT_XLSX = Path(os.getenv("OUTPUT_XLSX", "отзывы_мебели_АНАЛИЗ.xlsx"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
RUN_GPT = os.getenv("RUN_GPT", "0") == "1"
GPT_SAMPLE_SIZE = int(os.getenv("GPT_SAMPLE_SIZE", "30"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
SENT_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$", re.S)

CATEGORY_WORDS = {
    "service": "продавец менеджер консультант обслуживание сервис персонал магазин салон дозвон горячая линия договор помощь компетентн груб вежлив",
    "quality": "качество качеств качествен брак сломал сломалась сломался механизм рассыпал некомплект дефект скрипит хрупк прочн материал сборк",
    "delivery": "доставк достав привез привезл привезут привозит курьер груз срок сборщик",
    "description": "соответств описан",
    "return_claim": "возврат вернуть возвратил возвратят претензи гаранти обмен замена рекламац расторг деньги",
    "price": "цен дорог дорого дорогов скидк стоим рассрочк дешев",
    "soft": "диван кресл кроват пуф матрас мягк уголок тахт банкетк",
    "cabinet": "шкаф стол комод тумб кухн прихож стенк полк пенал гардероб стеллаж письменн кроватка",
}
POSITIVE = "отличн хорош довольн рекоменд прекрасн замечательн понрав приятн благодар спасибо супер качествен удобн быстро оперативн вовремя идеальн".split()
NEGATIVE = "ужасн отвратительн плох хренов разочарован недовольн кошмар проблем задерж опоздал долг долго брак сломал некомплект недостав некачествен груб хам обман нервотрепк".split()


def normalize_token(token: str) -> str:
    token = token.lower()
    if MORPH is not None:
        return MORPH.parse(token)[0].normal_form
    return STEMMER.stem(token)


def tokens(text: str) -> list[str]:
    return [normalize_token(t) for t in TOKEN_RE.findall(str(text))]


CATEGORY_TERMS = {
    key: {normalize_token(w) for w in value.split()} for key, value in CATEGORY_WORDS.items()
}
POS_TERMS = {normalize_token(w) for w in POSITIVE}
NEG_TERMS = {normalize_token(w) for w in NEGATIVE}


def has_category(text: str, category: str) -> bool:
    raw = str(text).lower()
    if category == "description":
        patterns = [
            r"не\s+соответств",
            r"соответств(?:ует|ие)\s+.*\bне\b",
            r"не\s+так(?:ой|ая|ое|ие)\s+(?:как|на)",
            r"не\s+совпада",
            r"отлича(?:ется|ется)\s+от\s+(?:описан|фото|каталог)",
            r"цвет\s+(?:не\s+тот|другой)",
            r"размер\s+(?:не\s+тот|другой)",
            r"комплектац(?:ия|ии)\s+не\s+соответ",
            r"на\s+фото\s+.*(?:друг|не\s+так)",
            r"описани(?:е|я)\s+.*(?:не\s+полно|не\s+соответ|не\s+совпад)",
            r"размер(?:ы|а)?\s+.*(?:не\s+тот|другой|не\s+соответ)",
            r"комплект(?:ация|а)?\s+.*(?:не\s+полн|не\s+тот|другой)",
        ]
        return any(re.search(p, raw) for p in patterns)
    return bool(set(tokens(raw)) & CATEGORY_TERMS[category])


def delivery_speed_flag(text: str, speed: str) -> bool:
    delivery_terms = CATEGORY_TERMS["delivery"]
    if speed == "fast":
        speed_words = "быстро быстр оперативно оперативн скоро скор вовремя своевременно своеврем".split()
    else:
        speed_words = "долго долг задержка задерж задержали задерживал опоздал опоздание просроч просрочка поздно ожидал ожидание".split()
    speed_terms = {normalize_token(w) for w in speed_words}
    for sentence in SENT_RE.findall(str(text).lower()):
        st = set(tokens(sentence))
        if st & delivery_terms and st & speed_terms:
            return True
    return False


def sentiment(text: str) -> tuple[float, str]:
    ts = tokens(text)
    if not ts:
        return 0.0, "нейтральный"
    pos = sum(t in POS_TERMS for t in ts)
    neg = sum(t in NEG_TERMS for t in ts)
    score = (pos - neg) / len(ts)
    label = "положительный" if score > 0.015 else "отрицательный" if score < -0.015 else "нейтральный"
    return round(float(score), 4), label


def chi2_binary(df: pd.DataFrame, flag: str) -> dict:
    table = pd.crosstab(df[flag].astype(int), df["positive_rating"].astype(int))
    if table.shape != (2, 2):
        return {"chi2": np.nan, "p": np.nan, "n": int(df[flag].sum())}
    stat, p, _, _ = chi2_contingency(table)
    return {"chi2": float(stat), "p": float(p), "n": int(df[flag].sum())}


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_excel(INPUT_XLSX, sheet_name=0)
    required = ["Наименование", "Оценка автора", "Дата", "Текст"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Входной файл не содержит поля: {missing}")

    df = df[required].copy()
    df = df.rename(columns={"Наименование": "company", "Оценка автора": "rating", "Текст": "text"})
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[(df["text"] != "") & df["rating"].notna()].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["Дата"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["negative"] = (df["rating"] <= 3).astype(int)
    df["positive_rating"] = (df["rating"] >= 4).astype(int)

    for cat in CATEGORY_WORDS:
        df[cat] = df["text"].map(lambda x, c=cat: int(has_category(x, c)))
    df["fast_delivery"] = df["text"].map(lambda x: int(delivery_speed_flag(x, "fast")))
    df["slow_delivery"] = df["text"].map(lambda x: int(delivery_speed_flag(x, "slow")))
    df["soft_only"] = ((df["soft"] == 1) & (df["cabinet"] == 0)).astype(int)
    df["cab_only"] = ((df["cabinet"] == 1) & (df["soft"] == 0)).astype(int)
    df["words"] = df["text"].str.split().str.len()
    df["length"] = df["text"].str.len()
    df["excl"] = df["text"].str.count("!")
    df["question"] = df["text"].str.count(r"\?")
    sent = df["text"].map(sentiment)
    df["sentiment_score"] = sent.map(lambda x: x[0])
    df["sentiment_label"] = sent.map(lambda x: x[1])
    df["sentiment_intensity"] = df["sentiment_score"].abs()
    return df


def run_gpt_validation(df: pd.DataFrame) -> pd.DataFrame | None:
    api_key = os.getenv("YANDEX_GPT_API_KEY")
    folder_id = os.getenv("YANDEX_CLOUD_FOLDER_ID")
    if not api_key or not folder_id:
        if RUN_GPT:
            raise RuntimeError("RUN_GPT=1, но не заданы YANDEX_GPT_API_KEY и YANDEX_CLOUD_FOLDER_ID")
        return None

    import requests

    sample = df[["company", "rating", "text"]].sample(n=min(GPT_SAMPLE_SIZE, len(df)), random_state=42)
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    rows = []
    for idx, row in sample.iterrows():
        prompt = (
            "Разметь отзыв для аналитики. Верни только JSON с ключами "
            "service, quality, delivery, description_mismatch, return_claim, "
            "soft_furniture, cabinet_furniture, sentiment. Значения категорий 0/1, "
            "sentiment: positive/neutral/negative. Не считай само слово 'описание' доказательством несоответствия. "
            f"Отзыв: {row['text'][:4000]}"
        )
        payload = {
            "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0},
            "messages": [
                {"role": "system", "text": "Ты аналитик клиентских отзывов."},
                {"role": "user", "text": prompt},
            ],
        }
        response = requests.post(
            url,
            headers={"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        rows.append({"row_id": int(idx), "text": row["text"], "gpt_response": json.dumps(response.json(), ensure_ascii=False)})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "gpt_validation_sample.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    df = load_and_prepare()

    complaint_cols = {
        "Сервис": "service",
        "Качество мебели": "quality",
        "Доставка": "delivery",
        "Соответствие описанию": "description",
        "Цена": "price",
    }
    for old, new in {"service": "Сервис", "delivery": "Доставка", "quality": "Качество мебели", "description": "Соответствие описанию", "return_claim": "Возврат/претензия", "price": "Цена"}.items():
        df[new] = df[old]
    df["Месяц"] = df["month"]
    df["Дата"] = df["date"]
    df["Оценка автора"] = df["rating"].astype(int)
    df["Текст"] = df["text"]
    df["Наименование"] = df["company"]

    rating_dist = df["rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="reviews")
    complaint_freq = pd.DataFrame({"topic": list(complaint_cols), "mentions": [int(df[v].sum()) for v in complaint_cols.values()]})
    company = (
        df.groupby("company", dropna=False)
        .agg(reviews=("rating", "size"), avg_rating=("rating", "mean"), negative_share=("negative", "mean"), delivery_mentions=("delivery", "sum"), service_mentions=("service", "sum"), quality_mentions=("quality", "sum"))
        .reset_index().sort_values(["reviews", "avg_rating"], ascending=[False, False])
    )
    monthly = (
        df.dropna(subset=["month"]).groupby("month")
        .agg(reviews=("rating", "size"), avg_rating=("rating", "mean"), negative_share=("negative", "mean"))
        .reset_index()
    )

    neg = df[df["negative"] == 1]
    neg_service = int(neg["service"].sum())
    neg_quality = int(neg["quality"].sum())
    b = int(((neg["service"] == 1) & (neg["quality"] == 0)).sum())
    c = int(((neg["service"] == 0) & (neg["quality"] == 1)).sum())
    h1_p = float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue) if b + c else np.nan
    h1_verdict = "Подтверждена" if neg_service > neg_quality and h1_p < 0.05 else "Не подтверждена"

    fast = df[df["fast_delivery"] == 1]
    slow = df[df["slow_delivery"] == 1]
    fast_mean = float(fast.rating.mean()) if len(fast) else np.nan
    slow_mean = float(slow.rating.mean()) if len(slow) else np.nan
    fast_pos = float(fast.positive_rating.mean()) if len(fast) else np.nan
    slow_pos = float(slow.positive_rating.mean()) if len(slow) else np.nan
    h2_p = float(mannwhitneyu(fast.rating, slow.rating, alternative="two-sided").pvalue) if len(fast) and len(slow) else np.nan
    h2_chi = chi2_binary(df, "fast_delivery")
    h2_verdict = "Подтверждена по текстовому прокси" if h2_p < 0.05 and fast_mean > slow_mean else "Не подтверждена"

    # H3: only cause categories inside return/claim reviews; the umbrella flag itself is excluded.
    claims = df[df["return_claim"] == 1]
    cause_cols = {k: v for k, v in complaint_cols.items() if k != "Возврат/претензия"}
    h3_counts = {k: int(claims[v].sum()) for k, v in cause_cols.items()}
    top_h3 = max(h3_counts, key=h3_counts.get) if h3_counts else None
    h3_verdict = "Подтверждена" if top_h3 == "Соответствие описанию" else "Не подтверждена"

    soft = df[df["soft_only"] == 1]
    cab = df[df["cab_only"] == 1]
    h4_p = float(mannwhitneyu(soft.words, cab.words, alternative="two-sided").pvalue) if len(soft) and len(cab) else np.nan
    h4_int_p = float(mannwhitneyu(soft.sentiment_intensity, cab.sentiment_intensity, alternative="two-sided").pvalue) if len(soft) and len(cab) else np.nan
    h4_verdict = "Подтверждена" if (soft.words.mean() > cab.words.mean() and h4_p < 0.05 and soft.sentiment_intensity.mean() >= cab.sentiment_intensity.mean()) else "Не подтверждена"
    h5_verdict = "Невозможно проверить"

    hypotheses = pd.DataFrame([
        ["H1", "Сервис vs качество в негативных", len(neg), neg_service, neg_quality, h1_p, h1_verdict],
        ["H2", "Скорость доставки и оценка", len(fast) + len(slow), fast_mean, slow_mean, h2_p, h2_verdict],
        ["H3", "Причина возвратов/претензий", len(claims), h3_counts.get("Соответствие описанию", 0), h3_counts.get(top_h3, 0) if top_h3 else 0, np.nan, h3_verdict],
        ["H4", "Мягкая vs корпусная мебель", len(soft) + len(cab), float(soft.words.mean()), float(cab.words.mean()), h4_p, h4_verdict],
        ["H5", "Цена и подробность отзыва", len(df), np.nan, np.nan, np.nan, h5_verdict],
    ], columns=["hypothesis", "metric", "coverage_n", "group_a", "group_b", "p_value", "verdict"])

    gpt_result = run_gpt_validation(df)
    gpt_status = {
        "requested": RUN_GPT,
        "credentials_present": bool(os.getenv("YANDEX_GPT_API_KEY") and os.getenv("YANDEX_CLOUD_FOLDER_ID")),
        "executed": gpt_result is not None,
        "sample_size": 0 if gpt_result is None else len(gpt_result),
        "normalizer": NORMALIZER,
    }
    (OUTPUT_DIR / "gpt_status.json").write_text(json.dumps(gpt_status, ensure_ascii=False, indent=2), encoding="utf-8")

    export_cols = ["Наименование", "Оценка автора", "Дата", "Текст", "Сервис", "Доставка", "Качество мебели", "Соответствие описанию", "Возврат/претензия", "Цена", "fast_delivery", "slow_delivery", "soft_only", "cab_only", "length", "words", "excl", "question", "Месяц", "sentiment_score", "sentiment_intensity", "sentiment_label"]
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df[export_cols].to_excel(writer, sheet_name="Отзывы", index=False)
        df[df["Дата"].notna()][export_cols].to_excel(writer, sheet_name="Отзывы_с_датой", index=False)
        rating_dist.to_excel(writer, sheet_name="Распределение_оценок", index=False)
        complaint_freq.to_excel(writer, sheet_name="Частотность_жалоб", index=False)
        company.to_excel(writer, sheet_name="Компании", index=False)
        monthly.to_excel(writer, sheet_name="Динамика_по_месяцам", index=False)
        hypotheses.to_excel(writer, sheet_name="Гипотезы", index=False)
        pd.DataFrame([
            {"Параметр": "Всего отзывов", "Значение": len(df)},
            {"Параметр": "С датой", "Значение": int(df.date.notna().sum())},
            {"Параметр": "Без даты", "Значение": int(df.date.isna().sum())},
            {"Параметр": "Покрытие H2", "Значение": round((len(fast) + len(slow)) / len(df), 4)},
            {"Параметр": "Покрытие H4", "Значение": round((len(soft) + len(cab)) / len(df), 4)},
            {"Параметр": "Нормализация текста", "Значение": NORMALIZER},
            {"Параметр": "YandexGPT выполнен", "Значение": gpt_status["executed"]},
        ]).to_excel(writer, sheet_name="Контроль_качества", index=False)

    summary = {
        "n": len(df), "with_date": int(df.date.notna().sum()), "without_date": int(df.date.isna().sum()),
        "h1": {"negative_n": len(neg), "service": neg_service, "quality": neg_quality, "p_value": h1_p, "verdict": h1_verdict},
        "h2": {"fast_n": len(fast), "slow_n": len(slow), "fast_mean": fast_mean, "slow_mean": slow_mean, "fast_positive_share": fast_pos, "slow_positive_share": slow_pos, "mannwhitney_p": h2_p, "chi2_fast_p": h2_chi["p"], "verdict": h2_verdict},
        "h3": {"claims_n": len(claims), "cause_counts": h3_counts, "top_cause": top_h3, "verdict": h3_verdict},
        "h4": {"soft_n": len(soft), "cab_n": len(cab), "soft_words": float(soft.words.mean()), "cab_words": float(cab.words.mean()), "words_p": h4_p, "sentiment_intensity_p": h4_int_p, "verdict": h4_verdict},
        "h5": {"verdict": h5_verdict},
        "gpt": gpt_status,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
