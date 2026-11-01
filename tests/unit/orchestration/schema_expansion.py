"""Independent test expansion of the local, acyclic planning schema references."""


def expand_schema(schema, value=None, seen=()):
    value = schema if value is None else value
    if isinstance(value, dict):
        if "$ref" in value:
            assert set(value) == {"$ref"}
            ref = value["$ref"]
            assert ref.startswith("#/$defs/") and ref not in seen
            return expand_schema(schema, schema["$defs"][ref[len("#/$defs/"):]], (*seen, ref))
        return {k: expand_schema(schema, v, seen) for k, v in value.items() if k != "$defs"}
    if isinstance(value, (list, tuple)):
        return tuple(expand_schema(schema, v, seen) for v in value)
    return value
