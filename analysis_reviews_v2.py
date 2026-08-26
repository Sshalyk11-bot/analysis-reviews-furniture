from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, binomtest
from nltk.stem.snowball import RussianStemmer

INPUT_XLSX = Path(os.getenv('INPUT_XLSX', 'отзывы_мебели_DataLens_ЧИСТЫЙ.xlsx'))
OUTPUT_XLSX = Path(os.getenv('OUTPUT_XLSX', 'отзывы_мебели_АНАЛИЗ_v2.xlsx'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'outputs'))
OUTPUT_DIR.mkdir(exist_ok=True)

stemmer = RussianStemmer()

# Normalized stems. Matching is done on tokens/stems, never by raw substring.
CATEGORY_WORDS = {
    'service': 'продавец менеджер консультант обслуживание сервис персонал магазин салон дозвон горячая линия договор помощь компетентн груб вежлив',
    'quality': 'качество качеств качествен брак сломал сломалась сломался механизм рассыпал некомплект дефект скрипит скрипит хрупк прочн материал сборк',
    'delivery': 'доставк достав привез привезл привезут привозит курьер груз срок сборщик',
    'description': 'соответств описан',
    'return_claim': 'возврат вернуть возвратил возвратят претензи гаранти обмен замена рекламац расторг деньги',
    'price': 'цен дорог дорого дорогов скидк стоим рассрочк дешев',
    'soft': 'диван кресл кроват пуф матрас мягк уголок тахт банкетк',
    'cabinet': 'шкаф стол комод тумб кухн прихож стенк полк пенал гардероб стеллаж письменн кроватка',
}
CATEGORY_STEMS = {k: {stemmer.stem(w) for w in v.split()} for k, v in CATEGORY_WORDS.items()}

POSITIVE = 'отличн хорош довольн рекоменд прекрасн замечательн понрав приятн благодар спасибо супер качествен удобн быстро оперативн вовремя идеальн'.split()
NEGATIVE = 'ужасн отвратительн плох хренов разочарован недовольн кошмар проблем задерж опоздал долг долго брак сломал некомплект недостав некачествен груб хам обман нервотрепк'.split()
POS_STEMS = {stemmer.stem(w) for w in POSITIVE}
NEG_STEMS = {stemmer.stem(w) for w in NEGATIVE}

TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.I)
SENT_RE = re.compile(r'[^.!?]+[.!?]+|[^.!?]+$', re.S)


def tokens(text: str) -> list[str]:
    return [stemmer.stem(t.lower()) for t in TOKEN_RE.findall(str(text))]


def has_category(text: str, category: str) -> bool:
    if category == 'description':
        t = str(text).lower()
        patterns = [
            r'не\s+соответств',
            r'соответств(?:ует|ие)\s+.*\bне\b',
            r'не\s+соответствующ',
            r'не\s+так(?:ой|ая|ое|ие)\s+(?:как|на)',
            r'не\s+совпада',
            r'отлича(?:ется|ется)\s+от\s+(?:описан|фото|каталог)',
            r'цвет\s+(?:не\s+тот|другой)',
            r'размер\s+(?:не\s+тот|другой)',
            r'комплектац(?:ия|ии)\s+не\s+соответ',
            r'на\s+фото\s+.*(?:друг|не\s+так)',
        ]
        return any(re.search(p, t) for p in patterns)
    ts = set(tokens(text))
    return bool(ts & CATEGORY_STEMS[category])


def sentence_tokens(sentence: str) -> set[str]:
    return set(tokens(sentence))


def delivery_speed_flag(text: str, speed: str) -> bool:
    """True only when a delivery term and a speed/slow term occur in the same sentence."""
    delivery_stems = CATEGORY_STEMS['delivery']
    if speed == 'fast':
        speed_words = 'быстро быстр оперативно оперативн скоро скор вовремя своевременно своеврем'.split()
    else:
        speed_words = 'долго долг задержка задерж задержали задерживал опоздал опоздание просроч просрочка поздно ожидал ожидание'.split()
    speed_stems = {stemmer.stem(w) for w in speed_words}
    for sent in SENT_RE.findall(str(text).lower()):
        st = sentence_tokens(sent)
        if st & delivery_stems and st & speed_stems:
            return True
    return False


def sentiment(text: str) -> tuple[float, str]:
    ts = tokens(text)
    if not ts:
        return 0.0, 'нейтральный'
    pos = sum(t in POS_STEMS for t in ts)
    neg = sum(t in NEG_STEMS for t in ts)
    score = (pos - neg) / max(len(ts), 1)
    if score > 0.015:
        label = 'положительный'
    elif score < -0.015:
        label = 'отрицательный'
    else:
        label = 'нейтральный'
    return round(float(score), 4), label


def chi2_binary(df: pd.DataFrame, flag: str, positive='positive_rating') -> dict:
    x = pd.crosstab(df[flag].astype(int), df[positive].astype(int))
    if x.shape != (2, 2):
        return {'chi2': np.nan, 'p': np.nan, 'n': int(df[flag].sum())}
    stat, p, dof, expected = chi2_contingency(x)
    return {'chi2': float(stat), 'p': float(p), 'n': int(df[flag].sum())}


def main():
    # Single source of truth: the exact workbook used for BI. Recompute every derived metric here.
    df = pd.read_excel(INPUT_XLSX, sheet_name='Отзывы')
    df = df.rename(columns={'Текст': 'text', 'Оценка автора': 'rating', 'Наименование': 'company'})
    df['text'] = df['text'].fillna('').astype(str).str.strip()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df[(df['text'] != '') & df['rating'].notna()].copy()
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df['Дата'], errors='coerce')
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    df['negative'] = (df['rating'] <= 3).astype(int)
    df['positive_rating'] = (df['rating'] >= 4).astype(int)

    # Transparent, reproducible categorization.
    for cat in ['service','quality','delivery','description','return_claim','price','soft','cabinet']:
        df[cat] = df['text'].map(lambda x, c=cat: int(has_category(x, c)))
    df['fast_delivery'] = df['text'].map(lambda x: int(delivery_speed_flag(x, 'fast')))
    df['slow_delivery'] = df['text'].map(lambda x: int(delivery_speed_flag(x, 'slow')))
    df['soft_only'] = ((df['soft'] == 1) & (df['cabinet'] == 0)).astype(int)
    df['cab_only'] = ((df['cabinet'] == 1) & (df['soft'] == 0)).astype(int)
    df['words'] = df['text'].str.split().str.len()
    df['length'] = df['text'].str.len()
    df['excl'] = df['text'].str.count('!')
    df['question'] = df['text'].str.count(r'\?')
    sent = df['text'].map(sentiment)
    df['sentiment_score'] = sent.map(lambda x: x[0])
    df['sentiment_label'] = sent.map(lambda x: x[1])
    df['sentiment_intensity'] = df['sentiment_score'].abs()

    # Compatibility names for BI / existing charts.
    df['Сервис'] = df['service']
    df['Доставка'] = df['delivery']
    df['Качество мебели'] = df['quality']
    df['Соответствие описанию'] = df['description']
    df['Возврат/претензия'] = df['return_claim']
    df['Цена'] = df['price']
    df['Месяц'] = df['month']
    df['Дата'] = df['date']
    df['Оценка автора'] = df['rating'].astype(int)
    df['Текст'] = df['text']
    df['Наименование'] = df['company']

    # Primary analysis.
    rating_dist = df['rating'].value_counts().sort_index().rename_axis('rating').reset_index(name='reviews')
    complaint_cols = {'Сервис':'service','Качество мебели':'quality','Доставка':'delivery','Соответствие описанию':'description','Возврат/претензия':'return_claim','Цена':'price'}
    complaint_freq = pd.DataFrame({'topic': list(complaint_cols), 'mentions': [int(df[c].sum()) for c in complaint_cols.values()]})
    company = (df.groupby('company', dropna=False)
               .agg(reviews=('rating','size'), avg_rating=('rating','mean'), negative_share=('negative','mean'),
                    delivery_mentions=('delivery','sum'), service_mentions=('service','sum'), quality_mentions=('quality','sum'))
               .reset_index().sort_values(['reviews','avg_rating'], ascending=[False,False]))
    monthly = (df.dropna(subset=['month']).groupby('month')
               .agg(reviews=('rating','size'), avg_rating=('rating','mean'), negative_share=('negative','mean'))
               .reset_index())

    # Hypothesis 1: use a statistically testable, mutually exclusive comparison within negatives.
    neg = df[df['negative'] == 1].copy()
    neg_service = int(neg['service'].sum())
    neg_quality = int(neg['quality'].sum())
    # McNemar is ideal for paired overlap, but we report descriptive difference and chi-square on presence indicators.
    h1_table = pd.crosstab(neg['service'], neg['quality'])
    b = int(((neg['service'] == 1) & (neg['quality'] == 0)).sum())
    c = int(((neg['service'] == 0) & (neg['quality'] == 1)).sum())
    h1_p = float(binomtest(min(b,c), n=b+c, p=0.5, alternative='two-sided').pvalue) if (b+c) else np.nan
    h1_verdict = 'Подтверждена' if neg_service > neg_quality and h1_p < 0.05 else 'Не подтверждена'

    # Hypothesis 2: delivery + speed in the SAME sentence; coverage explicit.
    fast = df[df['fast_delivery'] == 1]
    slow = df[df['slow_delivery'] == 1]
    fast_mean = float(fast.rating.mean()) if len(fast) else np.nan
    slow_mean = float(slow.rating.mean()) if len(slow) else np.nan
    fast_pos = float(fast.positive_rating.mean()) if len(fast) else np.nan
    slow_pos = float(slow.positive_rating.mean()) if len(slow) else np.nan
    h2_p = float(mannwhitneyu(fast.rating, slow.rating, alternative='two-sided').pvalue) if len(fast) and len(slow) else np.nan
    h2_chi = chi2_binary(df, 'fast_delivery')
    h2_verdict = 'Подтверждена по текстовому прокси' if h2_p < 0.05 and fast_mean > slow_mean else 'Не подтверждена'

    # Hypothesis 3: return/claim topics.
    claims = df[df['return_claim'] == 1]
    h3_counts = {k:int(claims[v].sum()) for k,v in complaint_cols.items()}
    top_h3 = max(h3_counts, key=h3_counts.get) if h3_counts else None
    h3_verdict = 'Не подтверждена' if top_h3 != 'Соответствие описанию' else 'Подтверждена'

    # Hypothesis 4: mutually exclusive product groups; coverage and sentiment model.
    soft = df[df['soft_only'] == 1]
    cab = df[df['cab_only'] == 1]
    h4_p = float(mannwhitneyu(soft.words, cab.words, alternative='two-sided').pvalue) if len(soft) and len(cab) else np.nan
    h4_excl_p = float(mannwhitneyu(soft.excl, cab.excl, alternative='two-sided').pvalue) if len(soft) and len(cab) else np.nan
    h4_int_p = float(mannwhitneyu(soft.sentiment_intensity, cab.sentiment_intensity, alternative='two-sided').pvalue) if len(soft) and len(cab) else np.nan
    h4_verdict = 'Подтверждена' if (soft.words.mean() > cab.words.mean() and h4_p < 0.05 and soft.sentiment_intensity.mean() >= cab.sentiment_intensity.mean()) else 'Не подтверждена'

    # Hypothesis 5: cannot test because no actual numeric product price.
    h5_verdict = 'Невозможно проверить'

    hypotheses = pd.DataFrame([
        ['H1','Сервис vs качество в негативных',len(neg),neg_service,neg_quality,h1_p,h1_verdict],
        ['H2','Скорость доставки и оценка',len(fast)+len(slow),fast_mean,slow_mean,h2_p,h2_verdict],
        ['H3','Причина возвратов/претензий',len(claims),h3_counts.get('Соответствие описанию',0),h3_counts.get(top_h3,0) if top_h3 else 0,np.nan,h3_verdict],
        ['H4','Мягкая vs корпусная мебель',len(soft)+len(cab),float(soft.words.mean()),float(cab.words.mean()),h4_p,h4_verdict],
        ['H5','Цена и подробность отзыва',len(df),np.nan,np.nan,np.nan,h5_verdict],
    ], columns=['hypothesis','metric','coverage_n','group_a','group_b','p_value','verdict'])

    # Save analysis tables.
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        export_cols = ['Наименование','Оценка автора','Дата','Текст','Сервис','Доставка','Качество мебели','Соответствие описанию','Возврат/претензия','Цена','fast_delivery','slow_delivery','soft_only','cab_only','length','words','excl','question','Месяц','sentiment_score','sentiment_intensity','sentiment_label']
        df[export_cols].to_excel(writer, sheet_name='Отзывы', index=False)
        df[df['Дата'].notna()][export_cols].to_excel(writer, sheet_name='Отзывы_с_датой', index=False)
        rating_dist.to_excel(writer, sheet_name='Распределение_оценок', index=False)
        complaint_freq.to_excel(writer, sheet_name='Частотность_жалоб', index=False)
        company.to_excel(writer, sheet_name='Компании', index=False)
        monthly.to_excel(writer, sheet_name='Динамика_по_месяцам', index=False)
        hypotheses.to_excel(writer, sheet_name='Гипотезы', index=False)
        pd.DataFrame([{'Параметр':'Всего отзывов', 'Значение':len(df)}, {'Параметр':'С датой', 'Значение':int(df.date.notna().sum())}, {'Параметр':'Без даты', 'Значение':int(df.date.isna().sum())}, {'Параметр':'Покрытие H2', 'Значение':round((len(fast)+len(slow))/len(df),4)}, {'Параметр':'Покрытие H4', 'Значение':round((len(soft)+len(cab))/len(df),4)}]).to_excel(writer, sheet_name='Контроль_качества', index=False)

    # Plots.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5)); plt.bar(rating_dist.rating.astype(str), rating_dist.reviews); plt.xlabel('Оценка'); plt.ylabel('Количество отзывов'); plt.title('Распределение отзывов по оценкам'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'01_rating_distribution.png', dpi=160); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(complaint_freq.topic, complaint_freq.mentions); plt.xticks(rotation=30, ha='right'); plt.ylabel('Упоминания'); plt.title('Частотность проблемных тематик'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'02_complaints.png', dpi=160); plt.close()
    plt.figure(figsize=(10,5)); plt.plot(monthly.month, monthly.reviews, marker='o'); plt.ylabel('Количество отзывов'); plt.title('Динамика количества отзывов по месяцам'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'03_monthly_reviews.png', dpi=160); plt.close()
    top_company = company.head(15).sort_values('negative_share'); plt.figure(figsize=(9,6)); plt.barh(top_company.company.astype(str), top_company.negative_share*100); plt.xlabel('Негативные отзывы, %'); plt.title('Доля негатива по компаниям (топ-15 по числу отзывов)'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'04_company_negative_share.png', dpi=160); plt.close()
    plt.figure(figsize=(7,5)); plt.bar(['Быстрая доставка','Проблемная доставка'], [fast_pos*100, slow_pos*100]); plt.ylabel('Оценки 4–5, %'); plt.title('Связь текстовых признаков доставки с оценкой'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'05_delivery_positive_share.png', dpi=160); plt.close()
    plt.figure(figsize=(7,5)); plt.bar(['Мягкая мебель','Корпусная мебель'], [soft.words.mean(), cab.words.mean()]); plt.ylabel('Среднее число слов'); plt.title('Длина отзывов по типу мебели'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'06_furniture_length.png', dpi=160); plt.close()
    plt.figure(figsize=(8,5)); plt.bar(hypotheses.hypothesis, hypotheses.coverage_n); plt.ylabel('Количество отзывов'); plt.title('Покрытие проверок гипотез'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'07_hypothesis_coverage.png', dpi=160); plt.close()
    plt.figure(figsize=(7,5)); pd.crosstab(df['sentiment_label'], df['positive_rating']).plot(kind='bar'); plt.ylabel('Количество отзывов'); plt.title('Лексиконная тональность и фактическая оценка'); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'08_sentiment_vs_rating.png', dpi=160); plt.close()

    # Optional YandexGPT integration. It is deliberately disabled unless credentials are provided.
    gpt_note = {
        'enabled': bool(os.getenv('YANDEX_GPT_API_KEY') and os.getenv('YANDEX_CLOUD_FOLDER_ID')),
        'message': 'Set YANDEX_GPT_API_KEY and YANDEX_CLOUD_FOLDER_ID to run the optional LLM validation sample.'
    }
    (OUTPUT_DIR/'gpt_status.json').write_text(json.dumps(gpt_note, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {
        'n': len(df), 'with_date': int(df.date.notna().sum()), 'without_date': int(df.date.isna().sum()),
        'h1': {'negative_n':len(neg),'service':neg_service,'quality':neg_quality,'chi2_p':h1_p,'verdict':h1_verdict},
        'h2': {'fast_n':len(fast),'slow_n':len(slow),'coverage':(len(fast)+len(slow))/len(df),'fast_mean':fast_mean,'slow_mean':slow_mean,'fast_positive_share':fast_pos,'slow_positive_share':slow_pos,'mannwhitney_p':h2_p,'chi2_fast_p':h2_chi['p'],'verdict':h2_verdict},
        'h3': {'claims_n':len(claims),'counts':h3_counts,'verdict':h3_verdict},
        'h4': {'soft_n':len(soft),'cab_n':len(cab),'coverage':(len(soft)+len(cab))/len(df),'soft_words':float(soft.words.mean()),'cab_words':float(cab.words.mean()),'words_p':h4_p,'excl_p':h4_excl_p,'soft_sentiment_intensity':float(soft.sentiment_intensity.mean()),'cab_sentiment_intensity':float(cab.sentiment_intensity.mean()),'sentiment_intensity_p':h4_int_p,'verdict':h4_verdict},
        'h5': {'verdict':h5_verdict},
        'top_companies': company.head(10).to_dict('records'),
    }
    (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

if __name__ == '__main__':
    main()

# --- Optional YandexGPT validation helper (not used for the numeric hypothesis tests) ---
def yandex_gpt_classify_sample(texts: list[str], api_key: str | None = None, folder_id: str | None = None):
    """Classify a small sample with YandexGPT when credentials are supplied.

    Environment variables: YANDEX_GPT_API_KEY and YANDEX_CLOUD_FOLDER_ID.
    The LLM output is stored separately and is not mixed with the deterministic
    metrics, so the main analysis remains reproducible.
    """
    import requests
    api_key = api_key or os.getenv('YANDEX_GPT_API_KEY')
    folder_id = folder_id or os.getenv('YANDEX_CLOUD_FOLDER_ID')
    if not api_key or not folder_id:
        raise RuntimeError('Set YANDEX_GPT_API_KEY and YANDEX_CLOUD_FOLDER_ID')
    url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'
    results = []
    for text in texts:
        prompt = ('Определи для отзыва категории: сервис, качество мебели, доставка, '
                  'несоответствие описанию, возврат/претензия, мягкая/корпусная мебель. '
                  'Также дай тональность positive/neutral/negative. Ответ только JSON. Отзыв: ' + text[:4000])
        payload = {'modelUri': f'gpt://{folder_id}/yandexgpt/latest', 'completionOptions': {'stream': False, 'temperature': 0},
                   'messages': [{'role': 'system', 'text': 'Ты аналитик отзывов.'}, {'role': 'user', 'text': prompt}]}
        r = requests.post(url, headers={'Authorization': f'Api-Key {api_key}', 'Content-Type':'application/json'}, json=payload, timeout=60)
        r.raise_for_status()
        results.append({'text': text, 'gpt_response': r.json()})
    return pd.DataFrame(results)
