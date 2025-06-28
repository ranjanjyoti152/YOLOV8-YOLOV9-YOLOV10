# Contributing to YOLO Training Project

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Coding Standards

- Use Python 3.8+ features and type hints
- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Include comprehensive docstrings for all functions and classes
- Use logging instead of print statements for debugging
- Handle exceptions gracefully with proper error messages

### Testing

- Write tests for new functionality
- Ensure all tests pass before submitting PR
- Include both unit tests and integration tests
- Test with different Python versions (3.8-3.11)

### Code Style

We use several tools to maintain code quality:

```bash
# Install development dependencies
pip install flake8 black isort pytest

# Format code
black .
isort .

# Lint code
flake8 .

# Run tests
pytest
```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

### Bug Reports

Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

### Feature Requests

We welcome feature requests! Please provide:

- Clear description of the feature
- Use case and motivation
- Any implementation ideas you might have

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md)
