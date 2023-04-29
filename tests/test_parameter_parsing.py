"""Test user parameter parsing."""
import pytest
import os
import shutil
import logging
import src.material_topopt.utilities as utils


#######################################################################################################################
#######################################################################################################################
def test_user_parameter_parsing_correct():
    try:
        shutil.rmtree('output')
    except:
        print("No directory 'output' to remove.")
    logging_level = logging.ERROR
    relative_logfile_path = os.path.join('output', 'logfile.txt')
    utils.setup_logging(logging_level, logfile_path=relative_logfile_path)

    user_parameters = \
        {
            "param_int": 10,
            "param_float": 0.5,
            "param_string": "my string"
        }

    value = utils.get_parameter_or_default(user_parameters, "param_int", int, bounds = (0, 10),
                                           default_value = None, required = True, additional_message = "Integer")
    assert value == 10

    value = utils.get_parameter_or_default(user_parameters, "param_float", float, bounds = (0.0, 0.5),
                                           default_value = None, required = True, additional_message = "Float")
    assert value == 0.5

    value = utils.get_parameter_or_default(user_parameters, "param_string", str, bounds = (None, None),
                                           default_value = None, required = True, additional_message = "String")
    assert value == "my string"

    value = utils.get_parameter_or_default(user_parameters, "param_else", str, bounds = (None, None),
                                           default_value = None, required = False, additional_message = "Other")
    assert value is None

#######################################################################################################################
#######################################################################################################################
def test_user_parameter_parsing_incorrect():
    user_parameters = \
        {
            "param_int": 10,
            "param_float": 0.5,
            "param_string": "my string"
        }

    with pytest.raises(TypeError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_int", float, bounds = (0, 10),
                                       default_value = None, required = True, additional_message = "Integer")
    assert "Integer" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_int", int, bounds = (0, 5),
                                       default_value = None, required = True, additional_message = "Integer")
    assert "required to be less than or equal to 5" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_int", int, bounds = (15, 25),
                                       default_value = None, required = True, additional_message = "Integer")
    assert "required to be greater than or equal to 15" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_float", float, bounds = (None, 0.25),
                                       default_value = None, required = True, additional_message = "Float")
    assert "required to be less than or equal to 0.25" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_int", float, bounds = (None, 0.25),
                                       default_value = None, required = True, additional_message = "Integer")
    assert "Integer" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_else", float, bounds = (None, 0.25),
                                       default_value = None, required = True, additional_message = "Required")
    assert "Required" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_else", float, bounds = (None, 0.25),
                                       default_value = 1, required = True, additional_message = "Required")
    assert "Required" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_string", str, bounds = (0, 0.25),
                                       default_value = None, required = True, additional_message = "String")

    with pytest.raises(TypeError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_string", str, bounds = (None, 0.25),
                                       default_value = None, required = True, additional_message = "String")

    with pytest.raises(ValueError) as excinfo:
        utils.get_parameter_or_default(user_parameters, "param_float", float, bounds = (1.0, 0.0),
                                       default_value = None, required = True, additional_message = "Float")

