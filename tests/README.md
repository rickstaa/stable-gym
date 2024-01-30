# Unit Tests

This directory contains unit tests for the environments found in the [stable-gym](https://github.com/rickstaa/stable-gym) repository. These tests are crucial to ensure that our translated environments behave as expected and align with the original sources they are based on.

## Testing Framework

We use the [pytest](https://docs.pytest.org/en/latest/) framework for our unit tests. Pytest is a mature full-featured Python testing tool that helps you write better programs.

## Running the Tests

To run the unit tests, navigate to the directory containing the tests and execute the following command in your terminal:

```bash
pytest
```

This command will automatically discover and run all the tests in the directory.

## Test Results

After running the command, pytest will provide a detailed report of the tests that passed and those that failed, along with the corresponding error messages if any.

Remember, passing tests ensure that changes to the environments do not break existing functionality. Always make sure all tests pass after making changes to the environments.
