"""Permission table and permission-gated command executor (Task 4).

The executor does NOT verify identity itself. It receives the output of the
verification pipeline (identity + whether it was verified) and decides.
"""
from dataclasses import dataclass

# role -> set of allowed commands
PERMISSIONS = {
    "admin":      {"unlock", "delete", "lights_on", "lights_off",
                   "status", "stop"},
    "user":       {"lights_on", "lights_off", "status", "stop"},
    "unverified": set(),
}

# spoken word -> action
WORD_TO_COMMAND = {
    "on": "lights_on",
    "off": "lights_off",
    "go": "status",
    "stop": "stop",
    "yes": "unlock",
    "no": "delete",
}

# identity -> role (edit names to match your enrolled team members)
USER_ROLES = {
    "basmala": "admin",
     "mohamed": "user",
     "ahmed": "user",
}


@dataclass
class Result:
    executed: bool
    command: str | None
    message: str


def role_of(identity: str | None, verified: bool) -> str:
    """Unverified or unknown identities get the 'unverified' role."""
    if not verified or identity is None:
        return "unverified"
    return USER_ROLES.get(identity, "user")


def execute(word: str | None, identity: str | None, verified: bool) -> Result:
    """Run a command only if the verified identity has permission."""
    if word is None:
        return Result(False, None, "Command not understood.")

    command = WORD_TO_COMMAND.get(word)
    if command is None:
        return Result(False, None, f"Unknown word '{word}'.")

    role = role_of(identity, verified)
    if command not in PERMISSIONS[role]:
        who = identity if verified else "unverified speaker"
        return Result(False, command,
                      f"DENIED: {who} (role={role}) cannot '{command}'.")

    return Result(True, command, f"OK: '{command}' executed for {identity}.")