def sort_lists(dicts: list[dict], *list_keys: str) -> None:
    for d in dicts:
        for k in list_keys:
            list_value = d[k]
            if list_value:
                list_value.sort()
