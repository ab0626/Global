"""A subset of the GDELT DOC 2.0 query language, evaluated over the local tables.

Supported: bare terms, "quoted phrases", `-negation`, `OR` groups (bare or
parenthesised) and the field operators below. Unsupported GDELT operators are
reported back to the caller instead of being silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

FIELD_OPERATORS = {
    "domain",
    "domainis",
    "sourcelang",
    "sourcecountry",
    "theme",
    "location",
    "actor",
    "quadclass",
}
TOKEN_PATTERN = re.compile(r'\(|\)|-?[A-Za-z]+:"[^"]*"|"[^"]*"|[^\s()]+')
COMPARISON_PATTERN = re.compile(r"^[A-Za-z]+[<>]")


class QueryError(ValueError):
    """Raised for a query the API cannot evaluate."""


@dataclass(frozen=True)
class Term:
    field: str | None
    value: str

    def matches(self, columns: set[str]) -> pl.Expr:
        value = self.value.lower()
        if self.field is None:
            return pl.col("searchtext").str.contains(value, literal=True)
        if self.field == "domain":
            return pl.col("domain").str.to_lowercase().str.contains(value, literal=True)
        if self.field == "domainis":
            return pl.col("domain").str.to_lowercase() == value
        if self.field == "sourcelang":
            language = pl.col("language").str.to_lowercase()
            return (language == value) | language.str.starts_with(value[:3])
        if self.field == "sourcecountry":
            return pl.col("sourcecountry").str.to_lowercase().str.contains(value, literal=True)
        if self.field == "quadclass":
            if not value.isdigit():
                raise QueryError(f"quadclass expects 1-4, got {self.value!r}")
            return pl.col("quadclasses").list.contains(int(value))
        list_column = {"theme": "themes", "location": "locations", "actor": "actors"}[self.field]
        if list_column not in columns:
            raise QueryError(f"{self.field} is not available on this collection")
        return (
            pl.col(list_column)
            .list.eval(pl.element().str.to_lowercase().str.contains(value, literal=True))
            .list.any()
        )


@dataclass(frozen=True)
class Group:
    """One AND-ed slot of the query; several terms inside it are OR-ed."""

    terms: tuple[Term, ...]
    negate: bool

    def matches(self, columns: set[str]) -> pl.Expr:
        expression = self.terms[0].matches(columns)
        for term in self.terms[1:]:
            expression = expression | term.matches(columns)
        return ~expression if self.negate else expression


def parse_term(token: str) -> tuple[Term, bool]:
    negate = token.startswith("-")
    token = token[1:] if negate else token
    if COMPARISON_PATTERN.match(token):
        raise QueryError(f"comparison operators such as {token!r} are not served")
    field = None
    if ":" in token and not token.startswith('"'):
        candidate, _, rest = token.partition(":")
        candidate = candidate.lower()
        if candidate in FIELD_OPERATORS:
            field, token = candidate, rest
        elif candidate not in {"http", "https"}:
            raise QueryError(f"unsupported operator {candidate!r}")
    value = token.strip('"').strip()
    if not value:
        raise QueryError("empty search term")
    return Term(field, value), negate


def parse(query: str) -> list[Group]:
    tokens = TOKEN_PATTERN.findall(query or "")
    groups: list[Group] = []
    pending: list[Term] = []
    pending_negate = False
    in_parentheses = False
    expect_or = False

    def flush() -> None:
        nonlocal pending, pending_negate, expect_or
        if pending:
            groups.append(Group(tuple(pending), pending_negate))
        pending, pending_negate, expect_or = [], False, False

    for token in tokens:
        if token == "(":
            flush()
            in_parentheses = True
            continue
        if token == ")":
            in_parentheses = False
            flush()
            continue
        if token.upper() == "OR":
            if not pending:
                raise QueryError("OR must sit between two terms")
            expect_or = True
            continue
        term, negate = parse_term(token)
        if pending and not (expect_or or in_parentheses):
            flush()
        if not pending:
            pending_negate = negate
        pending.append(term)
        expect_or = False
    flush()
    return groups


def filter_frame(frame: pl.DataFrame, query: str) -> pl.DataFrame:
    groups = parse(query)
    if not groups:
        return frame
    columns = set(frame.columns)
    expression = groups[0].matches(columns)
    for group in groups[1:]:
        expression = expression & group.matches(columns)
    return frame.filter(expression)
