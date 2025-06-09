# 🌱 Rotifer: A Genetic AI That Evolves Itself  
## 🧠 Autonomous Neural Evolution | Faster Convergence Than Keras  
**"The most powerful AI is one that builds itself."** - Mahdi Forghani Fard, 2023

**Our brain wasn't designed by an engineer. Why should our AIs be?**

 [![PHP](https://img.shields.io/badge/Built%20With-PHP-blue.svg)](#)

Rotifer is a cutting-edge **Genetic Machine Learning (AutoML) Framework** that **evolves its neural network architecture, layers, and weights** - all without manual intervention.

Unlike Keras or traditional frameworks where architecture is fixed, **Rotifer builds itself from scratch**, adding/removing neurons and connections via genetic algorithms. This results in having the most optimized and a truly emergent structure for the training dataset, and **faster convergence** - like nature intended.

## 🔥 Why Use Rotifer?

- 🧬 **Auto-evolving neural networks** (AutoML)
- 🚀 **Faster convergence** than traditional fixed-structure models
- 🧠 **No manual layer or neuron count tuning.** Possible to set, otherwise it evolves to find the best setup
- 🔄 **Mutation, crossover, and self-replication**
- 💡 **Ideal for creative AI and problem solving without predefined architecture**
- 🧩 Examples in the repository: **Autoencoder**, **Memory-Based learning**, and **XOR**
- 💻 Written in **pure PHP** - lightweight and hackable


## 🧬 How It Works
Rotifer simulates a digital world of agents. Each agent has some genes which define a neural network with weights. The framework handles:

- Neurons
- Connections
- Mutation and crossover
- Activation functions
- Selection and reproduction

Over generations, the population evolves better architectures, learning the task without ever knowing the number of layers or neurons required.

![Single Layer neural network with intra-connections](https://github.com/mahdyfo/php-genetic-ai-automl/blob/main/neural_layerings.jpg?raw=true)

These 2 neural networks are identical. All hidden layers can be combined into a single layer with intra-connections.

Unlike traditional networks with rigid layers, Rotifer condenses complexity into a single, dynamically growing layer. This way we eliminate the need for manual configuration of neuron and layer counts. This single hidden-layer gets very complex and not understandable for humans after several generations of evolution by genetic algorithm. This is not important for us because we don't want to analyze them. We just want to make the network powerful, and indeed it will be very powerful.

## 📦 Install

```bash
git clone https://github.com/mahdyfo/rotifer.git

cd rotifer

composer install
```

## 🧪 Built-in Examples
Task|Run File|Description
----|--------|---------------
🧠 XOR|xor.php|Classic XOR learning in few generations
🧠 Memory|memory.php|Remembers sequence outputs even if inputs repeat
🧠 AutoEncoder|autoencoder.php|Compress and decompress data through evolved layers
🧠 Other|yourfile.php|Write any other script you want using the examples


## 🚀 Quick Start
✅ Solve XOR in seconds:
```bash
php xor.php
```
Example output:
```
Generation 1 - Best generation fitness: 5.3965296271639 - Best overall fitness: 5.3965296271639
Generation 50 - Best generation fitness: 5.9992278738651 - Best overall fitness: 5.9992278738651
Generation 100 - Best generation fitness: 6.0455893609229 - Best overall fitness: 6.7389574321586
Generation 150 - Best generation fitness: 7.4842880310069 - Best overall fitness: 7.6137585607025
Generation 200 - Best generation fitness: 7.5486734099125 - Best overall fitness: 7.9401862706856

Report:
  Best fitness => 7.940186270685596 (value based on the fitness function)
  Hidden Neurons Count => 7 (The hidden neurons created automatically for the best agent after 200 generations of evolution)
  Connections Count => 52 (The total connections count between neurons. Not all neurons are connected which results in being precise, fast and powerful)
  
Test:
    Rounded Output: 1 - Raw output: 0.99712500243069
    Rounded Output: 0 - Raw output: 0.00030062252549047
    Rounded Output: 0 - Raw output: 0.0019566823546141
    Rounded Output: 1 - Raw output: 0.99504714984784
    Rounded Output: 1 - Raw output: 0.99970922413458
    Rounded Output: 0 - Raw output: 0.00004324516515
    Rounded Output: 0 - Raw output: 0.0042442188361674
    Rounded Output: 1 - Raw output: 0.95487230539894
(You see how close are the numbers to the intended outputs? Try it with Keras to see the difference!)
```

## 📚 Code Breakdown
Create world and agents:
```php
$population = 100;
// XOR data with bias value
$data = [
    [[1, 0, 0], [0]],
    [[1, 0, 1], [1]],
    [[1, 1, 0], [1]],
    [[1, 1, 1], [0]],
    //^ Here we set the first value of each input as bias (1). You can skip adding bias by removing them
];
$inputs = 3; // Inputs dimension
$outputs = 1; // Outputs dimension
$hasMemory = false; // Whether it should remember the sequence or not. For xor it shouldn't, for LLM it should.

$layers = []; // Empty, to auto-find the optimal layers and neurons structure by evolution
$layers = [3, 2]; // Or not empty, to use fixed layers count. 3 neurons in 1st hidden layer and 2 neurons in the 2nd hidden layer. This way, the evolution only finds weights.
// If using dynamic-layers, set PROBABILITY_MUTATE_* constants to values more than zero

$world = new World('MyExampleWorld');
$world->createAgents($population, $inputs, $outputs, $layers, $hasMemory);
```
---
Fitness function:
```php
$fitnessFunction = function (Agent $agent, $dataRow, $otherAgents, $world) {
    // Here you have access to important variables such as the current $agent, the current inputs and outputs in $dataRow,
    // Other agents and the world instance. So you can even make the agents communicate with each other!
    $predicted = $agent->getOutputValues()[0]; // The predicted value based on the current agent genes
    $actual = $dataRow[1][0]; // The training set actual value
    return 1.0 - abs($predicted - $actual); // fitness function. Opposite of error function, the higher is the better
};
```
---
Run the world:
```php
$generations = 30; // How many generations you want the world simulation to continue? 0 to make it endless
$survivalRate = 0.2; // 20% of the best agents in the world can reproduce the next generation. Like our actual world (hypergamy)
$world->step($fitnessFunction, $data, $generations, $survivalRate);
```
---
Test best agent:
```php
$agent = $world->getBestAgent();
$agent->reset(); // Reset its memory, like a new-born child. If it has any memory from training phase.
// Iterate through data and test the inputs one by one to see output values
foreach ($data as $row) {
    $agent->step($row[0]);
    echo $agent->getOutputValues()[0];
}
```
---
Load saved worlds:
```php
// Load the saved world to continue the training process
$world = World::loadAutoSaved('MyExampleWorld');
```
---
Constants:
```php
const PROBABILITY_CROSSOVER = 0.5; // Crossover probability, 0.5 mean half genes from mother and half from father
const PROBABILITY_MUTATE_WEIGHT = 0.4; // Percentage of agents to get mutated
const MUTATE_WEIGHT_COUNT = 1; // number of weight mutations in every agent that gets mutated
const PROBABILITY_MUTATE_ADD_NEURON = 0.04; // Possibility of adding a neuron in a mutation
const PROBABILITY_MUTATE_REMOVE_NEURON = 0.04; // Possibility of removing a neuron in a mutation
const PROBABILITY_MUTATE_ADD_GENE = 0.1; // Possibility of adding a new connection between neurons in a mutation
const PROBABILITY_MUTATE_REMOVE_GENE = 0.1; // Possibility of removing a connection between two neurons in a mutation
const ACTIVATION = [Activation::class, 'sigmoid']; // The activation function. Find more values in Activation class
const SAVE_WORLD_EVERY_GENERATION = 0; // 0 means don't save
const CALCULATE_STEP_TIME = false; // calculate each epoch time of the agent and saves in $agent->stepTime
const ONLY_CALCULATE_FIRST_STEP_TIME = false; // Doesn't calculate all steps, but only the first step
```
---

## ❤️ Support & Contribution
We welcome issues, stars, pull requests, collaborations, feel free to contribute ideas or optimize the agent training strategies. Potential ideas:
- Parallelization
- GPU
- Visualizer for agent evolution
- Web interface for experiments
- CLI commands for training/testing
- Example for multi-agent cooperative environments

## ⭐️ Star Us
If you believe in self-evolving AI, please give this repo a ⭐️ to support future development and spread awareness!

genetic ai - automl - evolving neural network - genetic algorithm neural net - neural architecture search - php ai - machine learning - self-evolving ai - keras alternative - pytorch alternative - autoencoder genetic - xor ai

