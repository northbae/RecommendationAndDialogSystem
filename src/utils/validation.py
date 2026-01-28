from typing import List, Optional, Tuple
import pandas as pd


def validate_article_ids(
        input_str: str,
        similarity_df: pd.DataFrame,
        allow_empty: bool = False
) -> Tuple[Optional[List[int]], Optional[str]]:

    if not input_str or input_str.strip() == '':
        if allow_empty:
            return [], None
        else:
            return None, "Не введено ни одного ID"

    try:
        input_str = input_str.replace(',', ' ')
        ids = [int(x.strip()) for x in input_str.split() if x.strip()]

        if len(ids) != len(set(ids)):
            unique_ids = list(dict.fromkeys(ids))
            return unique_ids, f"Удалены дубликаты"

        valid_ids = []
        invalid_ids = []

        for aid in ids:
            if aid in similarity_df.index:
                valid_ids.append(aid)
            else:
                invalid_ids.append(aid)

        if invalid_ids:
            if not valid_ids:
                return None, f"Все ID не найдены: {invalid_ids}"
            else:
                return valid_ids, f"ID не найдены и пропущены: {invalid_ids}"

        return valid_ids, None

    except ValueError as e:
        return None, f"неверный формат ввода"


def validate_number(
        input_str: str,
        min_val: int = 1,
        max_val: int = 100,
        default: int = 10
) -> Tuple[int, Optional[str]]:

    if not input_str or input_str.strip() == '':
        return default, None

    try:
        num = int(input_str.strip())

        if num < min_val or num > max_val:
            return default, f"Число должно быть от {min_val} до {max_val}. Использовано: {default}"

        return num, None

    except ValueError:
        return default, f"Неверный формат. Использовано значение по умолчанию: {default}"


def validate_choice(
        input_str: str,
        valid_choices: List[str],
        default: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:

    input_str = input_str.strip()

    if not input_str:
        if default:
            return default, None
        else:
            return None, "Не сделан выбор"

    if input_str in valid_choices:
        return input_str, None
    else:
        return default, f"Неверный выбор. Допустимые: {valid_choices}"