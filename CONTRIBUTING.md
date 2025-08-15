# Contributing to AI Forex Trading Bot

Thank you for your interest in contributing to the AI Forex Trading Bot! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- MetaTrader 5 platform (for live trading)
- Basic understanding of forex trading concepts
- Familiarity with machine learning concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/ai-forex-trading-bot.git
   cd ai-forex-trading-bot
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Configure Environment**
   ```bash
   cp src/config.example.py src/config.py
   # Edit src/config.py with your settings
   ```

## 📝 Development Workflow

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests and Linting**
   ```bash
   # Run tests
   pytest tests/

   # Check code formatting
   black --check .

   # Run linting
   ruff check .

   # Type checking
   mypy src/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Pull Request Process

1. **Push Your Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Use a clear and descriptive title
   - Fill out the PR template
   - Link any related issues
   - Ensure all checks pass

3. **Review Process**
   - Address review feedback promptly
   - Keep discussions focused and constructive
   - Update your branch if needed

## 🧪 Testing Guidelines

### Test Types

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Sanity Tests**: Basic smoke tests for core functionality

### Test Requirements

- All new features must include tests
- Maintain or improve test coverage
- Tests should be independent and repeatable
- Use descriptive test names and docstrings

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=src tests/
```

## 📊 Performance Guidelines

### Trading Bot Specific

- **Backtesting**: Test strategies on historical data
- **Paper Trading**: Validate with demo accounts first
- **Risk Management**: Always implement proper risk controls
- **Performance Monitoring**: Include logging and metrics

### Code Performance

- Profile critical trading functions
- Optimize data processing pipelines
- Monitor memory usage with large datasets
- Consider async operations for I/O bound tasks

## 🏗️ Architecture Guidelines

### Project Structure

```
src/                    # Source code
├── main.py            # Main trading bot entry point
├── config.py          # Configuration settings
├── models/            # ML models and strategies
├── integrations/      # External API integrations
└── utils/             # Utility functions

tests/                 # Test files
docs/                  # Documentation
.github/workflows/     # CI/CD pipelines
```

### Design Principles

- **Modularity**: Keep components loosely coupled
- **Testability**: Design with testing in mind
- **Configurability**: Use configuration files/environment variables
- **Observability**: Include comprehensive logging
- **Security**: Never commit credentials or sensitive data

## 🔒 Security Guidelines

### Sensitive Data

- Never commit API keys, passwords, or account numbers
- Use environment variables for sensitive configuration
- Include `.env` files in `.gitignore`
- Use the `config.example.py` pattern for sharing configuration structure

### Trading Security

- Always test with demo accounts first
- Implement position size limits
- Use stop-loss mechanisms
- Monitor for unusual trading patterns

## 📚 Documentation

### Code Documentation

- Use clear and descriptive docstrings
- Follow Google or NumPy docstring style
- Document complex algorithms and trading logic
- Include examples for public APIs

### Project Documentation

- Update README.md for significant changes
- Maintain the architecture documentation
- Document configuration options
- Include troubleshooting guides

## 🐛 Reporting Issues

### Bug Reports

Include the following information:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant log output or error messages

### Feature Requests

- Describe the feature and its use case
- Explain the expected behavior
- Consider providing implementation suggestions
- Discuss potential impacts on existing functionality

## 💡 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Documentation**: Check the docs/ directory first
- **Code Review**: Ask for help in pull request comments

## 📈 Trading Disclaimers

### Important Notes

- **Educational Purpose**: This project is primarily for educational purposes
- **Financial Risk**: Trading involves significant financial risk
- **No Guarantees**: Past performance does not guarantee future results
- **Demo First**: Always test thoroughly with demo accounts
- **Personal Responsibility**: Users are responsible for their trading decisions

### Risk Management

- Never trade with money you can't afford to lose
- Always use proper position sizing
- Implement stop-loss mechanisms
- Monitor your bot's performance regularly
- Have a plan for unexpected market conditions

## 🎯 Roadmap and Priorities

### Current Focus Areas

1. **Stability**: Improving bot reliability and error handling
2. **Performance**: Optimizing ML model training and prediction speed
3. **Testing**: Expanding test coverage and automated testing
4. **Documentation**: Improving user guides and API documentation

### Future Enhancements

- Multi-asset trading support
- Advanced ML strategies
- Real-time market analysis
- Portfolio management features
- Mobile app integration

Thank you for contributing to the AI Forex Trading Bot! Your contributions help make this project better for everyone. 🚀
