"""Shared scripted provider identities; scientific fixtures live with their owners."""
from agent.orchestration import PlanningModelProfile
PROFILES=tuple(PlanningModelProfile(key,'scripted','model/'+key) for key in ('alpha','beta'))
