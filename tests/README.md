# Rotifer Test Suite

## Overview

Comprehensive testing suite for the Rotifer genetic AI framework that evolves its own neural network architecture.

## Test Structure

### Unit Tests (`tests/Unit/`)
Tests individual components in isolation:

#### Models
- **NeuronTest** - 12 tests for neuron creation, connections, activation functions
- **AgentTest** - 32 tests for agent creation, genome management, neural network operations
- **WorldTest** - 15 tests for population management, evolution, persistence

#### Helpers
- **WeightHelperTest** - 7 tests for random weight generation and bounds checking
- **ReproductionHelperTest** - 15 tests for crossover and mutation operations

#### Activations
- **ActivationTest** - 18 tests for sigmoid, ReLU, leaky ReLU, tanh, and threshold functions

### Integration Tests (`tests/Integration/`)
Tests component interactions:

- **GeneticEvolutionTest** - 9 tests for evolution process, fitness improvement, selection
- **GenomeEncodingTest** - 12 tests for encoding/decoding with Binary, Hex, Human, JSON encoders

### Functional Tests (`tests/Functional/`)
Tests complete workflows using real examples:

- **XOREvolutionTest** - 6 tests for XOR problem solving with static and dynamic agents
- **MemoryNetworkTest** - 8 tests for sequence learning and memory retention
- **AutoEncoderTest** - 7 tests for data compression through evolved bottleneck layers

## Running Tests

### Run All Tests
```bash
vendor/bin/phpunit
```

### Run Specific Test Suite
```bash
vendor/bin/phpunit --testsuite Unit
vendor/bin/phpunit --testsuite Integration
vendor/bin/phpunit --testsuite Functional
```

### Run with Detailed Output
```bash
vendor/bin/phpunit --testdox
```

### Run Specific Test File
```bash
vendor/bin/phpunit tests/Unit/Models/AgentTest.php
```

### Generate Code Coverage Report (requires Xdebug)
```bash
vendor/bin/phpunit --coverage-html coverage
```

## Test Results

**Total: 142 tests**
- Unit Tests: 99
- Integration Tests: 21
- Functional Tests: 22

**Coverage:**
- Core Models (Neuron, Agent, World): ✓
- Genetic Operations (Crossover, Mutation): ✓
- Activation Functions: ✓
- Evolution Process: ✓
- Genome Encoding: ✓
- XOR Learning: ✓
- Memory Networks: ✓
- AutoEncoders: ✓

## Key Test Scenarios

### 1. XOR Problem Solving
Tests demonstrate that evolved networks can learn the XOR function:
- Static architectures (fixed layers)
- Dynamic architectures (evolving neurons)
- Fitness improvement over generations
- Prediction accuracy

### 2. Memory Networks
Tests verify sequence learning capabilities:
- Memory persistence across steps
- Memory reset functionality
- Sequence pattern recognition
- Long-term memory behavior

### 3. AutoEncoder Learning
Tests validate compression/reconstruction:
- Bottleneck layer compression
- Multi-output reconstruction
- Different architecture configurations
- Identity mapping

### 4. Genetic Evolution
Tests ensure proper evolution mechanics:
- Fitness improvement over time
- Population diversity maintenance
- Survival selection
- Dynamic architecture evolution
- Crossover and mutation operations

## Test Constants

Tests use these constants (defined in test files):
- `ACTIVATION`: Sigmoid by default
- `PROBABILITY_CROSSOVER`: 0.5
- `PROBABILITY_MUTATE_WEIGHT`: 0.4
- `PROBABILITY_MUTATE_ADD_NEURON`: 0.04
- `PROBABILITY_MUTATE_ADD_GENE`: 0.1
- `SAVE_WORLD_EVERY_GENERATION`: 0 (disabled in tests)

## Notes

- Tests automatically clean up `autosave/` directory after execution
- Some functional tests may show evolution progress output
- Evolution-based tests may have slight variance due to randomness
- Tests validate both convergence and diversity in populations

## Continuous Integration

To integrate with CI/CD:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    composer install
    vendor/bin/phpunit --coverage-text
```

## Contributing

When adding new features:
1. Add unit tests for individual components
2. Add integration tests for component interactions
3. Add functional tests for end-to-end workflows
4. Ensure all tests pass before submitting PR
