""""""
from datetime import datetime
from pathlib import Path
from typing import *

DateLike = Union[str, datetime, datetime.date]
PathOrStr = Union[str, Path]
OptPathOrStr = Optional[Union[str, Path]]
OptSeq = Optional[Sequence]
