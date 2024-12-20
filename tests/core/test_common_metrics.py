import pytest
import os
import shutil


@pytest.mark.parametrize(
    "plot_config", [{"problem_type": "classification"}], indirect=True
)
def test_auc(plot_config, setup_and_teardown):
    pass
