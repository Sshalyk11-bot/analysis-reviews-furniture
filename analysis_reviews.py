import pandas as pd
import re
from scipy.stats import mannwhitneyu

df = pd.read_excel("reviews_yandex (1).xlsx")

# Удаляем отзывы без текста/оценки и дубликаты
d = df[df["Текст"].notna() & df["Оценка автора"].notna()].copy()
d = d.drop_duplicates(subset=["Текст"]).copy()
d["rating"] = pd.to_numeric(d["Оценка автора"], errors="coerce")
d["text"] = d["Текст"].astype(str).str.lower()

patterns = {
    "Сервис": r"менеджер|консультант|персонал|продав|сервис|обслужив|сотрудник|салон|оператор|связ[ьи]|дозвон|навяз|хам|груб|игнор|обещ",
    "Доставка": r"достав|привез|привезли|курьер|срок|опозд|задерж|ожидал|ждал|ждать|приех|доставщик|подъ[её]м",
    "Качество мебели": r"качеств|слом|сломал|сломался|сломана|брак|дефект|царап|скол|полом|каркас|механизм|обивк|ткан|фурнитур|дсп|фанер|поролон|скрип|продавил|развал",
    "Соответствие описанию": r"не соответств|не соответствует|не совпад|отлича[её]тся|на фото|по фото|фото|описан|описанию|вживую|другой цвет|цвет.*не|размер.*не|комплектац",
    "Возврат/претензия": r"возврат|вернут|вернуть|претенз|рекламац|деньги.*верн|верн.*деньг|обмен|гарант",
    "Цена": r"цен[ауы]|стоим|дорог|дешев|скидк|акци[яи]|рассроч|предоплат",
}
for name, pattern in patterns.items():
    d[name] = d["text"].str.contains(pattern, regex=True, na=False)

d["fast_delivery"] = d["text"].str.contains(
    r"быстр|оператив|скор[оа]|в кратчай|без задерж|раньше срок", regex=True
)
d["slow_delivery"] = d["text"].str.contains(
    r"долг|задерж|опозд|перенос|просроч|ждал|ждать|ожидал|ожидание|месяц.*достав|срок.*прош|не привез|не достав",
    regex=True
)
d["soft"] = d["text"].str.contains(
    r"диван|диваны|кресл|мягк(ая|ой|ую|ие)? мебель|матрас|пуф|тахт|оттоман|софа", regex=True
)
d["cabinet"] = d["text"].str.contains(
    r"шкаф|стол|комод|тумб|стенк|кухн|гардероб|прихож|полк|пенал|стеллаж|мебель.*корпус", regex=True
)
d["soft_only"] = d.soft & ~d.cabinet
d["cab_only"] = d.cabinet & ~d.soft
d["words"] = d["text"].str.split().str.len()
d["length"] = d["text"].str.len()
d["excl"] = d["text"].str.count("!")
d["positive_rating"] = d["rating"] >= 4

# Гипотеза 1
negative = d[d["rating"] <= 3]
print("Гипотеза 1:", negative["Сервис"].sum(), negative["Качество мебели"].sum())

# Гипотеза 2
fast = d[d.fast_delivery]
slow = d[d.slow_delivery]
print("Быстрая доставка:", fast.rating.mean(), fast.positive_rating.mean())
print("Проблемная доставка:", slow.rating.mean(), slow.positive_rating.mean())
print("Mann-Whitney p:", mannwhitneyu(
    fast.rating, d[~d.fast_delivery].rating, alternative="two-sided"
).pvalue)

# Гипотеза 3
claims = d[d["Возврат/претензия"]]
for c in ["Качество мебели","Сервис","Доставка","Цена","Соответствие описанию"]:
    print(c, claims[c].sum())

# Гипотеза 4
soft = d[d.soft_only]
cab = d[d.cab_only]
print("Среднее слов:", soft.words.mean(), cab.words.mean())
print("Mann-Whitney p:", mannwhitneyu(
    soft.words, cab.words, alternative="two-sided"
).pvalue)

# Гипотеза 5
print("Проверка невозможна: в исходном датасете нет числовой цены товара.")
