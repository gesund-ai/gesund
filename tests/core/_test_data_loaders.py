"""
Test suite for DataLoader in the gesund.core._data_loaders module.
"""

import pytest
from unittest.mock import mock_open, patch
from gesund.core._data_loaders import DataLoader, DataLoadError


@pytest.fixture
def data_loader():
    return DataLoader()


@pytest.fixture
def sample_json_data():
    return {"key": "value"}


@pytest.fixture
def valid_json_path():
    return "/valid/path/data.json"


@pytest.fixture
def invalid_json_path():
    return "/invalid/path/data.json"


@patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
@patch("json.load")
def test_json_loader_success(mock_json_load, mock_file, data_loader, sample_json_data, valid_json_path):
    mock_json_load.return_value = sample_json_data
    result = data_loader._json_loader(valid_json_path)
    mock_file.assert_called_once_with(valid_json_path, "r")
    mock_json_load.assert_called_once()
    assert result == sample_json_data


@patch("builtins.open", side_effect=FileNotFoundError)
def test_json_loader_file_not_found(mock_file, data_loader, invalid_json_path):
    with pytest.raises(DataLoadError) as exc_info:
        data_loader._json_loader(invalid_json_path)
    mock_file.assert_called_once_with(invalid_json_path, "r")
    assert str(exc_info.value) == "Could not load JSON file !"


@patch.object(DataLoader, "_json_loader")
def test_load_json(mock_json_loader, data_loader, valid_json_path, sample_json_data):
    mock_json_loader.return_value = sample_json_data
    result = data_loader.load(valid_json_path, "json")
    mock_json_loader.assert_called_once_with(valid_json_path)
    assert result == sample_json_data


def test_load_invalid_format(data_loader, valid_json_path):
    with pytest.raises(KeyError) as exc_info:
        data_loader.load(valid_json_path, "xml")
    assert "xml" in str(exc_info.value)
