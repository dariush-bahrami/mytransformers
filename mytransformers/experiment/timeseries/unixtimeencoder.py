import time
from typing import NamedTuple


class EncodedTimeTuple(NamedTuple):
    timestamp: int
    year: int
    month: int
    year_day: int
    month_day: int
    week_day: int
    hour: int
    minute: int
    second: int


def encode(timestamp: int) -> EncodedTimeTuple:
    time_struct = time.gmtime(timestamp)
    return EncodedTimeTuple(
        timestamp=timestamp,
        year=time_struct.tm_year,
        month=time_struct.tm_mon,
        year_day=time_struct.tm_yday,
        month_day=time_struct.tm_mday,
        week_day=time_struct.tm_wday,
        hour=time_struct.tm_hour,
        minute=time_struct.tm_min,
        second=time_struct.tm_sec,
    )


def decode(encoded_time: EncodedTimeTuple) -> int:
    return encoded_time.timestamp
