"""Exact historical publication references, distinct from intra-plan StepOutputRef."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PriorOutputBinding:
    run_id: str
    step_id: str
    tool_name: str
    source_port: str
    semantic_type: str
    accepted_step_sha256: str
    manifest_sha256: str
    authority_sha256: str
    verification_mode: str = 'scientific-authority.v2'

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not str or not value.strip():
                raise ValueError('Invalid prior-output identity.')
            if name.endswith('_sha256') and (len(value) != 64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Invalid prior-output digest.')
        if self.verification_mode != 'scientific-authority.v2':
            raise ValueError('Unsupported historical verification mode; no implicit authority upgrade.')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PriorOutputRef:
    binding: PriorOutputBinding
    output_key: str

    def __post_init__(self):
        if not isinstance(self.binding, PriorOutputBinding) or type(self.output_key) is not str or not self.output_key:
            raise ValueError('Invalid historical output member.')

    def to_dict(self):
        return {'$prior_output': {'binding': self.binding.to_dict(), 'output_key': self.output_key}}

    @classmethod
    def from_dict(cls, value):
        if set(value) != {'$prior_output'}:
            raise ValueError('Invalid historical reference shape.')
        record = value['$prior_output']
        if set(record) != {'binding', 'output_key'} or set(record['binding']) != set(PriorOutputBinding.__dataclass_fields__):
            raise ValueError('Invalid historical binding shape.')
        return cls(PriorOutputBinding(**record['binding']), record['output_key'])
