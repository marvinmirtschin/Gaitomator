import time

_NANOSECONDS_KEY = "nanoseconds"
_MICROSECONDS_KEY = "microseconds"
_MILLISECONDS_KEY = "milliseconds"
_SECONDS_KEY = "seconds"
_MINUTES_KEY = "minutes"
_HOURS_KEY = "hours"
_DAYS_KEY = "days"
_WEEKS_KEY = "weeks"


class MTime:
    CONVERSIONS = (1000, 1000, 1000, 60, 60, 24, 7)
    # Maybe save full name in classes too
    UNITS = (_NANOSECONDS_KEY, _MICROSECONDS_KEY, _MILLISECONDS_KEY, _SECONDS_KEY, _MINUTES_KEY, _HOURS_KEY, _DAYS_KEY, _WEEKS_KEY)
    SHORT_REPRESENTATIONS = {
        _NANOSECONDS_KEY : "ns",
        _MICROSECONDS_KEY: "us",
        _MILLISECONDS_KEY: "ms",
        _SECONDS_KEY     : "s",
        _MINUTES_KEY     : "m",
        _HOURS_KEY       : "h",
        _DAYS_KEY        : "d",
        _WEEKS_KEY       : "w"
    }

    def __init__(self, unit):
        self._unit = unit
        self.short_representation = self.SHORT_REPRESENTATIONS[unit]

    def to_nanoseconds(self, value):
        return self.to_(value, self.UNITS.index(_NANOSECONDS_KEY))

    def to_microseconds(self, value):
        return self.to_(value, self.UNITS.index(_MICROSECONDS_KEY))

    def to_milliseconds(self, value):
        return self.to_(value, self.UNITS.index(_MILLISECONDS_KEY))

    def to_seconds(self, value):
        return self.to_(value, self.UNITS.index(_SECONDS_KEY))

    def to_minutes(self, value):
        return self.to_(value, self.UNITS.index(_MINUTES_KEY))

    def to_hours(self, value):
        return self.to_(value, self.UNITS.index(_HOURS_KEY))

    def to_days(self, value):
        return self.to_(value, self.UNITS.index(_DAYS_KEY))

    def to_weeks(self, value):
        return self.to_(value, self.UNITS.index(_WEEKS_KEY))

    def to_(self, value, target_index):
        index = self.UNITS.index(self._unit)
        if index > target_index:
            for i in range(target_index, index):
                value *= self.CONVERSIONS[i]
        else:
            for i in range(index, target_index):
                value /= self.CONVERSIONS[i]
        return float(value)

    def from_unit(self, value, unit):
        return unit.to_(value, self.UNITS.index(self._unit))


NANOSECONDS = MTime(_NANOSECONDS_KEY)
MICROSECONDS = MTime(_MICROSECONDS_KEY)
MILLISECONDS = MTime(_MILLISECONDS_KEY)
SECONDS = MTime(_SECONDS_KEY)
MINUTES = MTime(_MINUTES_KEY)
HOURS = MTime(_HOURS_KEY)
DAYS = MTime(_DAYS_KEY)
WEEKS = MTime(_WEEKS_KEY)


def get_current_time(unit=None):
    time_in_s = time.time()
    if unit:
        unit_index = SECONDS.UNITS.index(unit)
        return SECONDS.to_(time_in_s, unit_index)
    else:
        return time_in_s


def test_conversion():
    assert SECONDS.from_unit(1000, MILLISECONDS) == 1
    assert SECONDS.from_unit(1000, NANOSECONDS) == 0.000001
    assert MILLISECONDS.from_unit(1000, MICROSECONDS) == 1
    assert SECONDS.from_unit(1, MINUTES) == 60
    assert DAYS.from_unit(24, HOURS) == 1
