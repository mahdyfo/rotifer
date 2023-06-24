# php-genetic-ai-automl
Evolutionary Genetic AI that designs itself (AutoML)

I believe the most powerful AI is the one which creates itself. A complex structure that human cannot understand, like our brains. 

Run:
```
php src/index.php --verbose
```

## Main Parts
**World**, **Agent**, **Neuron**, **Encoders**, **Activation functions**, **Reproduction**

## Layering

![Single Layer neural network with intra-connections](https://github.com/mahdyfo/php-genetic-ai-automl/blob/main/neural_layerings.jpg?raw=true)

These 2 neural networks are identical. All hidden layers can be combined into a single layer with intra-connections.

This way we eliminate the need for manual configuration of neuron and layer counts. This single hidden-layer gets very complex and not understandable for humans after several generations of evolution by genetic algorithm. This is not important for us because we don't want to analyze them. We just want to make the network powerful, and indeed it will be very powerful.

## Example:

```
$data = [
    [[0, 0, 0], [1]],
    [[0, 0, 1], [0]],
    [[0, 1, 0], [0]],
    [[0, 1, 1], [1]],
    [[1, 0, 0], [1]],
    [[1, 0, 1], [0]],
    [[1, 1, 0], [0]],
    [[1, 1, 1], [1]],
];

Generation 1 - Best generation fitness: 5.3965296271639 - Best overall fitness: 5.3965296271639
...
Generation 50 - Best generation fitness: 5.9992278738651 - Best overall fitness: 5.9992278738651
...
Generation 100 - Best generation fitness: 6.0455893609229 - Best overall fitness: 6.7389574321586
...
Generation 150 - Best generation fitness: 7.4842880310069 - Best overall fitness: 7.6137585607025
...
Generation 199 - Best generation fitness: 7.5486734099125 - Best overall fitness: 7.9401862706856
Generation 200 - Best generation fitness: 7.5486734099125 - Best overall fitness: 7.9401862706856

Report:
  Best fitness => 7.940186270685596
  Hidden Neurons Count => 7
  Connections Count => 52
  
Test:
    Rounded Output: 1 - Raw output: 0.99712500243069
    Rounded Output: 0 - Raw output: 0.00030062252549047
    Rounded Output: 0 - Raw output: 0.0019566823546141
    Rounded Output: 1 - Raw output: 0.99504714984784
    Rounded Output: 1 - Raw output: 0.99970922413458
    Rounded Output: 0 - Raw output: 6.5887410171281E-5
    Rounded Output: 0 - Raw output: 0.0042442188361674
    Rounded Output: 1 - Raw output: 0.95487230539894
```