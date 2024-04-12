import typing as tp

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[EntityScoreType]  # list of entity scores


def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)
    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    results = []
    for message in messages:
        entities_found = []
        # Простая логика для примера:
        # Если в сообщении есть "Сбер", добавляем сущность с ID 150 и оценкой 3.0
        if "Сбер" in message:
            entities_found.append((150, 3.0))
        # Если в сообщении есть "Тинькофф", добавляем сущность с ID 225 и оценкой 3.0
        if "Тинькофф" in message:
            entities_found.append((225, 3.0))

        # Изменение: Проверяем, есть ли найденные сущности, прежде чем добавлять список
        if entities_found:
            results.append(entities_found)
        else:
            results.append([()])  # Добавляем пустой список, если сущностей нет

    return results