"""This module was made to compare two tables and report the differences into Discord.

But the Discord has pretty heavy limits, so I had to save diffs in files and give access to you with
`2nd-circle.perchun.it`. Maybe, in the future, I will create a bot from this script (not just webhook).
But for now, this module is just some unused code.
"""
import dataclasses
import datetime
import typing as t

import numpy
import pandas

_T = t.TypeVar("_T")


@dataclasses.dataclass
class Change(t.Generic[_T]):
    old: _T
    new: _T

    @classmethod
    def only_if_changed(cls, old: _T, new: _T) -> t.Self | _T:
        """Only return self, if old and new are different."""
        if old == new:
            return old
        return cls(old, new)


@dataclasses.dataclass
class CircleData:
    application_deadline: datetime.datetime | Change[datetime.datetime] | None
    entrance_exam_date: str | datetime.datetime | Change[str | datetime.datetime] | None
    date_of_decision_if_test_isnt: datetime.datetime | Change[datetime.datetime] | None
    available_places: int | Change[int] | None

    @classmethod
    def from_dataframe(cls, data: pandas.Series, circle_num: t.Literal[2, 3, 4, 5, 6]) -> t.Self | None:
        offset = 0 if circle_num == 2 else 4 * (circle_num - 2)
        instance = cls(
            application_deadline=data[8 + offset],
            entrance_exam_date=data[9 + offset],
            date_of_decision_if_test_isnt=data[10 + offset],
            available_places=data[11 + offset],
        )

        return instance if any(filter(lambda x: x is not None, (
            getattr(instance, field.name)
            for field in dataclasses.fields(instance)
        ))) else None

    def compare_with_second_dataset(self, other: t.Self) -> t.Self:
        for field in dataclasses.fields(self):
            old = getattr(self, field.name)
            new = getattr(other, field.name)

            setattr(self, field.name, Change.only_if_changed(old, new))

        return self

    def is_changed(self) -> bool:
        return any(
            isinstance(getattr(self, field.name), Change)
            for field in dataclasses.fields(self)
        )


@dataclasses.dataclass
class ChangedSchool:
    id: float
    name: str | Change[str]
    address: str | Change[str] | None
    site: str | Change[str] | None
    profession_code: str | Change[str] | None
    profession_name: str | Change[str] | None
    second_circle: CircleData | None
    third_circle: CircleData | None
    fourth_circle: CircleData | None
    fifth_circle: CircleData | None
    sixth_circle: CircleData | None

    @classmethod
    def from_dataframe(cls, data: pandas.Series, previous: pandas.Series) -> t.Self:
        if data[0] is None:
            return  # naming row

        street = data[2] if data[2] is not None else "někde"
        address = str(street) + (f", {data[3]}" if data[3] is not None else "")

        return cls(
            id=data[0],
            name=data[1],
            address=address,
            site=data[4],
            profession_code=data[5],
            profession_name=data[6],
            second_circle=CircleData.from_dataframe(data, circle_num=2),
            third_circle=CircleData.from_dataframe(data, circle_num=3),
            fourth_circle=CircleData.from_dataframe(data, circle_num=4),
            fifth_circle=CircleData.from_dataframe(data, circle_num=5),
            sixth_circle=CircleData.from_dataframe(data, circle_num=6),
        )

    def compare_with_second_dataset(self, other: t.Self) -> t.Self:
        if self.id != other.id:
            raise ValueError("IDs are not the same")

        for field in dataclasses.fields(self):
            if field.name == "id":
                continue

            old = getattr(self, field.name)
            new = getattr(other, field.name)

            if isinstance(old, CircleData):
                setattr(self, field.name, old.compare_with_second_dataset(new))
            else:
                setattr(self, field.name, Change.only_if_changed(old, new))

        return self

    def is_changed(self) -> bool:
        return any(
            isinstance(getattr(self, field.name), Change)
            if not field.name.endswith("circle") or getattr(self, field.name) is None
            else getattr(self, field.name).is_changed()
            for field in dataclasses.fields(self)
        )


def diff_tables(table1_path: str, table2_path: str) -> list[ChangedSchool]:
    table1 = pandas.read_excel(table1_path).replace(numpy.nan, None)
    table2 = pandas.read_excel(table2_path).replace(numpy.nan, None)

    schools_in_first = transform_dataframe_to_objects(table1)
    schools_in_second = transform_dataframe_to_objects(table2)

    merged_schools = [
        old_school.compare_with_second_dataset(new_school)
        for old_school, new_school in zip(schools_in_first, schools_in_second)
    ]
    changed_schools = list(filter(lambda e: e.is_changed(), merged_schools))

    return changed_schools


def transform_dataframe_to_objects(table: pandas.DataFrame) -> list[ChangedSchool]:
    result: list[ChangedSchool] = []

    previous: t.Optional[pandas.Series] = None
    for i in range(2, len(table)):
        data = table.iloc[i]
        if data[0] is None:
            data[0:5] = previous[0:5]

        result.append(ChangedSchool.from_dataframe(data, previous))

        previous = data

    return result


if __name__ == "__main__":
    r = diff_tables("data/saves/private/excel/34.xlsx", "data/saves/private/excel/35.xlsx")
    print(r)
