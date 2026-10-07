"""Conservative scope checks, not a general semantic truth verifier."""

import re
from decimal import Decimal, InvalidOperation


def live_data_requirement(text: str) -> str | None:
    text = text.lower().replace("ё", "е")
    money = re.search(r"цен[аыуе]|стоимост|сколько\s+стоит", text)
    current = re.search(r"текущ|актуаль|сейчас|сегодня|назови|назовите|точн|сколько\s+стоит", text)
    stock = re.search(
        r"сколько.{0,80}(склад|остат|налич)|остат.{0,60}(склад|сейчас)|есть\s+ли.{0,60}в\s+налич",
        text,
    )
    payment = re.search(
        r"(поступил|поступила|поступление|статус).{0,30}(платеж|оплат)|оплачен[аы]?\s+(счет|заказ)",
        text,
    )
    status = re.search(r"(какой|текущий|покажи|узнай).{0,20}статус.{0,30}(заявк|заказ)", text)
    if money and current or stock or payment or status:
        return (
            "Актуальные цены, остатки, платежи и статусы нужно проверить в учётной системе. "
            "Этот сервис пока не подключён к API CRM и не может подтвердить эти значения."
        )
    return None


def numbers(text: str) -> set[Decimal]:
    text = re.sub(r"\[S\d+\]", "", text).replace("−", "-").replace("–", "-")
    # Grouped thousands and decimal separators; IDs in citations are excluded above.
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?:\D|$))", "", text)
    found = set()
    for value in re.findall(r"[+-]?\d+(?:[.,]\d+)?", text):
        try:
            found.add(Decimal(value.replace(",", ".")))
        except InvalidOperation:
            continue
    return found


def unsupported_numbers(text: str, evidence: str) -> set[Decimal]:
    return numbers(text) - numbers(evidence)
