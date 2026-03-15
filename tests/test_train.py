import os
import sys
import json
import pytest
import numpy as np

# Add project root / Ajouter la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from train import load_and_extract

def test_load_and_extract_valid_data(tmp_path):
    """Test loading valid data with mock conversations."""
    # Create mock data
    mock_data = [
        {
            "id": "conv_1",
            "label": 0,
            "label_name": "green",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "I feel okay"}
            ]
        },
        {
            "id": "conv_2",
            "label": 3,
            "label_name": "red",
            "messages": [
                {"role": "user", "content": "I want to die"},
                {"role": "assistant", "content": "I am here"},
                {"role": "user", "content": "Nothing matters"}
            ]
        }
    ]

    file_path = tmp_path / "mock_data.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    # Call the function
    X, y = load_and_extract(str(file_path))

    # Assertions
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 2  # 2 conversations
    assert X.shape[1] == 67 # 67 trajectory features
    assert y.shape == (2,)
    assert list(y) == [0, 3]

def test_load_and_extract_empty_data(tmp_path):
    """Test loading an empty list of conversations."""
    file_path = tmp_path / "empty_data.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    # Call the function
    X, y = load_and_extract(str(file_path))

    # Assertions
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 0
    assert y.shape == (0,)

def test_load_and_extract_missing_file():
    """Test loading a file that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_and_extract("non_existent_file.json")
