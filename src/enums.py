import enum

class RobotStates(enum.Enum):
    HIGH = 1
    LOW = -1

class HighActions(enum.Enum):
    SEARCH = 1
    WAIT = 0

class LowActions(enum.Enum):
    SEARCH = 1
    WAIT = 0
    RECHARGE = -1