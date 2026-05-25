"""Tests for customization regex parser."""
from __future__ import annotations

import pytest

from beverage_ai.aspects.customization import parse_customization_regex


@pytest.mark.parametrize("text, expected_sugar", [
    ("我点了三分糖", "三分"),
    ("无糖去冰", "无糖"),
    ("半糖", "五分"),
    ("七分糖正常冰", "七分"),
    ("全糖加波霸", "全糖"),
    ("正常糖", "全糖"),
    ("微糖去冰", "三分"),
])
def test_sugar_detection(text, expected_sugar):
    c = parse_customization_regex(text)
    assert c.sugar_level == expected_sugar


@pytest.mark.parametrize("text, expected_ice", [
    ("少冰", "少冰"),
    ("去冰", "无冰"),
    ("正常冰", "正常"),
    ("多冰", "多冰"),
    ("微冰", "少冰"),
    ("温的", "无冰"),
])
def test_ice_detection(text, expected_ice):
    c = parse_customization_regex(text)
    assert c.ice_level == expected_ice


def test_topping_detection():
    c = parse_customization_regex("加波霸 和 烧仙草")
    assert "珍珠" in c.toppings or "波霸" in str(c.toppings)
    assert "仙草" in c.toppings


def test_topping_dedup():
    """波霸 + 黑珍珠 should not both map to '珍珠'."""
    c = parse_customization_regex("我要黑珍珠 和 波霸 加芋圆")
    # The dedup keeps each canonical name once
    assert c.toppings.count("珍珠") <= 1


def test_size_detection():
    assert parse_customization_regex("大杯").size == "大"
    assert parse_customization_regex("中杯").size == "中"


def test_no_customization_returns_all_none():
    c = parse_customization_regex("好喝")
    assert c.sugar_level is None
    assert c.ice_level is None
    assert c.size is None
    assert c.toppings == []


def test_full_customization_text():
    c = parse_customization_regex("我点的是大杯七分糖少冰加芋圆加椰果")
    assert c.sugar_level == "七分"
    assert c.ice_level == "少冰"
    assert c.size == "大"
    assert "芋圆" in c.toppings
    assert "椰果" in c.toppings
