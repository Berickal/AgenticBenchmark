from dataclasses import dataclass
import sys
import os
sys.path.append("../")
from evaluation.metric import Metric, PassTest
import tempfile
import subprocess
import time
from dataclasses import dataclass

from unittest.mock import patch
from io import StringIO
import concurrent.futures
from evaluation.code_utils import clean_code, remove_duplicate

@dataclass
class TestReport:
    passed : bool
    functional_error : bool
    runtime_error : bool
    message : str

def update_test_name(response : str, test : str) -> str:
    """
    Update the test name in the response code.
    
    Args:
        response (str): The code response from the LLM.
        test (str): The test to be updated.
        
    Returns:
        str: The updated code with the new test name.
    """
    function_name = response.rsplit('def ', 1)[-1].split('(', 1)[0].strip()
    test_name = test.split("assert", 1)[1].split("(", 1)[0].strip()
    if function_name not in test:
        test = test.replace(test_name, function_name)
    return test

def evaluate_quixbugs_instance(response : str, 
                               tests : str, 
                               timeout : int = 10, 
                               programming_language : str = "python", 
                               ref_output : str = "", 
                               hard_clean : bool = False) -> TestReport:
    response = clean_code(response)
    if hard_clean:
        response = remove_duplicate(response, ref_output)

    if programming_language == "python":
        tests = update_test_name(response, tests)
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
            combined_code = f"{response}\n\n# Tests\n{tests}"
            temp_file.write(combined_code)
            temp_file_path = temp_file.name
        
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            if result.returncode == 0:
                return TestReport(
                    passed=True,
                    functional_error=False,
                    runtime_error=False,
                    message=f"All tests passed successfully.\nOutput: {result.stdout}"
                )
            else:
                if "AssertionError" in result.stderr or "assert" in result.stderr:
                    return TestReport(
                        passed=False,
                        functional_error=True,
                        runtime_error=False,
                        message=f"Test assertion failed: {result.stderr}"
                    )
                else:
                    return TestReport(
                        passed=False,
                        functional_error=False,
                        runtime_error=True,
                        message=f"Code execution error: {result.stderr}"
                    )
                    
        except subprocess.TimeoutExpired:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message="Test timed out"
            )
        except Exception as e:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message=f"Runtime error: {str(e)}"
            )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    else:
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message=f"Unsupported programming language: {programming_language}"
        )
    

def evaluate_mbbp_instance(response : str, tests : list[str], timeout : int = 10, ref_output : str = "") -> TestReport:
    """
    Evaluate a MBPP instance by running the provided code against the tests.
    
    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds.
        
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    return evaluate_quixbugs_instance(response, "\n".join(tests).replace('"None"', 'None'), timeout, programming_language="python", ref_output=ref_output)

def evaluate_human_eval_instance(response : str, tests : str, timeout : int = 10, ref_output : str = "") -> TestReport:
    """
    Evaluate a HumanEval instance by running the provided code against the tests.
    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds.
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    assert_tests = tests.split("def check(candidate):")[1]
    return evaluate_quixbugs_instance(response, assert_tests.replace("    ", ""), timeout, programming_language="python", ref_output=ref_output)




def run_code(code, test_input_lines):
    """
    Function to run the code with redirected input and output.
    """
    inputs_iter = iter(test_input_lines)
    fake_input = lambda: next(inputs_iter)
    fake_stdout = StringIO()

    with patch('builtins.input', fake_input), patch('sys.stdout', fake_stdout):
        exec(code, {})

    return fake_stdout.getvalue().strip()

def evaluate_condefect_instance(response : str, test_in : list[str], test_out : list[str], timeout : int = 10, ref_output : str = "", hard_clean : bool = False) -> TestReport:
    """
    Evaluate a ConDefect instance by running the provided code against the tests.
    
    Args:
        response (str): The code response from the LLM.
        tests (str): The test cases to be executed.
        timeout (int): Time limit for test execution in seconds.
        
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    response = clean_code(response)
    if hard_clean:
        response = remove_duplicate(response, ref_output)

    for t_in, t_out in zip(test_in, test_out):
        try:
            if not (os.path.exists(t_in) and os.path.exists(t_out)):
                raise FileNotFoundError(f"Test files {t_in} or {t_out} not found.")
            
            with open(t_in, 'r') as f:
                test_input_lines = f.readlines()
            
            with open(t_out, 'r') as f:
                test_output_lines = f.readlines()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_code, response, test_input_lines)
                try:
                    actual_output = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    return TestReport(
                        passed=False,
                        functional_error=False,
                        runtime_error=True,
                        message="Test timed out"
                    )
            expected_output = ''.join(test_output_lines).strip()
            if actual_output == expected_output:
                return TestReport(
                    passed=True,
                    functional_error=False,
                    runtime_error=False,
                    message="Test passed successfully."
                )
            else:
                return TestReport(
                    passed=False,
                    functional_error=True,
                    runtime_error=False,
                    message=f"Test failed: Expected '{expected_output}', got '{actual_output}'"
                )
        except Exception as e:
            return TestReport(
                passed=False,
                functional_error=False,
                runtime_error=True,
                message=f"Runtime error: {str(e)}"
            )
    

def evaluate_apps_instance(response: str, 
                        test_cases: dict, 
                        timeout: int = 10, 
                        programming_language: str = "python",
                        strict_whitespace: bool = False) -> TestReport:
    """
    Evaluate an APPS instance by running the provided code against input/output test cases.
    
    APPS format uses input/output pairs where:
    - inputs: list of input strings (stdin simulation)
    - outputs: list of expected output strings (stdout comparison)
    
    Args:
        response (str): The code response from the LLM.
        test_cases (dict): Dictionary with 'inputs' and 'outputs' lists.
        timeout (int): Time limit for test execution in seconds.
        programming_language (str): Programming language (currently only "python").
        strict_whitespace (bool): If False, normalizes trailing whitespace per line.
        
    Returns:
        TestReport: A report indicating whether the tests passed, and any errors encountered.
    """
    
    if programming_language != "python":
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message=f"Unsupported programming language: {programming_language}"
        )
    
    # Validate test case format
    if not isinstance(test_cases, dict) or 'inputs' not in test_cases or 'outputs' not in test_cases:
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message="Invalid test case format. Expected dict with 'inputs' and 'outputs' keys."
        )
    
    inputs = test_cases.get('inputs', [])
    expected_outputs = test_cases.get('outputs', [])
    
    if len(inputs) != len(expected_outputs):
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message=f"Mismatch between inputs ({len(inputs)}) and outputs ({len(expected_outputs)}) count."
        )
    
    if not inputs:
        return TestReport(
            passed=False,
            functional_error=True,
            runtime_error=False,
            message="No test cases provided."
        )
    
    # Create temporary file with the code
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as temp_file:
        temp_file.write(response)
        temp_file_path = temp_file.name
    
    try:
        # Test each input/output pair
        all_passed = True
        failed_cases = []
        
        for i, (test_input, expected_output) in enumerate(zip(inputs, expected_outputs)):
            try:
                # Run the code with the input
                result = subprocess.run(
                    [sys.executable, temp_file_path],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                if result.returncode != 0:
                    # Runtime error occurred
                    return TestReport(
                        passed=False,
                        functional_error=False,
                        runtime_error=True,
                        message=f"Runtime error in test case {i+1}: {result.stderr}"
                    )
                
                # Compare output with configurable whitespace handling
                actual_output = result.stdout.rstrip()
                expected_output_clean = expected_output.rstrip()
                
                if strict_whitespace:
                    # Strict comparison - exact match required
                    if actual_output == expected_output_clean:
                        continue  # Test passed
                else:
                    # Lenient comparison - normalize trailing spaces per line
                    actual_lines = [line.rstrip() for line in actual_output.split('\n')]
                    expected_lines = [line.rstrip() for line in expected_output_clean.split('\n')]
                    
                    if actual_lines == expected_lines:
                        continue  # Test passed with normalized whitespace
                
                # If comparison fails, mark as failed
                    all_passed = False
                    failed_cases.append({
                        'case': i + 1,
                        'input': test_input,
                        'expected': expected_output_clean,
                        'actual': actual_output
                    })
                    
            except subprocess.TimeoutExpired:
                return TestReport(
                    passed=False,
                    functional_error=False,
                    runtime_error=True,
                    message=f"Test case {i+1} timed out after {timeout} seconds"
                )
            except Exception as e:
                return TestReport(
                    passed=False,
                    functional_error=False,
                    runtime_error=True,
                    message=f"Unexpected error in test case {i+1}: {str(e)}"
                )
        
        if all_passed:
            return TestReport(
                passed=True,
                functional_error=False,
                runtime_error=False,
                message=f"All {len(inputs)} test cases passed successfully."
            )
        else:
            # Create detailed failure message
            failure_details = []
            for failed_case in failed_cases:
                if strict_whitespace:
                    case_detail = (
                        f"Test case {failed_case['case']} (strict whitespace):\n"
                        f"  Input: {repr(failed_case['input'])}\n"
                        f"  Expected: {repr(failed_case['expected'])}\n"
                        f"  Actual: {repr(failed_case['actual'])}"
                    )
                else:
                    case_detail = (
                        f"Test case {failed_case['case']} (normalized whitespace):\n"
                        f"  Input: {repr(failed_case['input'])}\n"
                        f"  Expected: {repr(failed_case['expected'])}\n"
                        f"  Actual: {repr(failed_case['actual'])}\n"
                    )
                failure_details.append(case_detail)
            
            failure_message = (
                f"Failed {len(failed_cases)} out of {len(inputs)} test cases:\n\n" +
                "\n\n".join(failure_details)
            )
            
            return TestReport(
                passed=False,
                functional_error=True,
                runtime_error=False,
                message=failure_message
            )
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)