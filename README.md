# Rotifer

**A genetic AI framework that evolves its own neural networks - modelled on how life actually evolves.**

Rotifer doesn't train networks with backpropagation. It *evolves* them: a population of organisms, each a neural network described entirely by its genome, competes and reproduces over generations. Topology, neuron count, and weights are all discovered automatically (AutoML / neuroevolution). On top of plain genetic search, Rotifer models the messy, powerful machinery of real evolution - **geographic islands, epigenetic trauma, self-tuning mutation, and lifetime learning that children inherit.**

Pure PHP. Watch it evolve live in your terminal **or** in a browser dashboard. Reproducible to the bit. Parallel across CPU cores.

```bash
composer install
php bin/rotifer run xor          # evolve XOR live in the terminal
php bin/rotifer list             # see all built-in problems
```

---

## Why it's different

| Traditional deep learning | Rotifer |
|---|---|
| Fixed architecture you design | Architecture is **discovered** by evolution |
| Gradient descent / backprop | Genetic operators: crossover + mutation |
| One global model | A **world of islands**, each its own gene pool |
| Weights are everything | The **genome is the network** - one array of connection genes |
| A black box | **Watch every generation** evolve, in terminal or browser |

## Core ideas

- **Genome = network.** A genome is just a list of connection genes (`from → to`, weight). There are no separate weight matrices; the genome *is* the network. (`src/Genome/`)
- **Organism.** A genome compiled into a runnable `Brain` plus the things evolution cares about - fitness, age, and an `Epigenome`. (`src/Organism/`)
- **World of islands.** The `World` runs several semi-isolated `Island`s ("villages"). Each evolves on its own and periodically migrates its best individuals to neighbours - spreading breakthroughs while preserving diversity. (`src/Evolution/`)
- **One seeded RNG tree.** Every random choice flows through a seedable `Rng`; the master seed derives an independent stream per island. **Same seed ⇒ identical run**, which makes evolution testable and parallel-safe. (`src/Runtime/Rng.php`)
- **Events, not print statements.** The engine emits events; *reporters* render them - a terminal dashboard, a JSON stream for the web UI, or nothing at all. (`src/Observe/`)

## The biology

Every mechanism is independently switchable in a problem's config; turned off, it's a no-op.

- **Epigenetic trauma** - hardship leaves a heritable, *decaying* stress marker that makes a lineage's offspring mutate harder for a few generations, then fades. Inherited trauma that washes out over time.
- **Adaptive mutation** - each island raises mutation when it stalls (explore) and lowers it when improving (exploit).
- **Lifetime learning** - an organism refines its own weights during its life (the Baldwin effect). A configurable fraction of what it learns is written back into its genome and **inherited** (Lamarckian).
- **Islands & migration** - different demes drift toward different solutions and trade their best on a ring.

## Run it

```bash
php bin/rotifer run xor                     # live terminal dashboard
php bin/rotifer run weather_forecast        # multi-class classification
php bin/rotifer run flappy_bird             # a game, learned with no training data
php bin/rotifer run xor --seed=42 --quiet   # reproducible, silent
php bin/rotifer run auto_encoder --parallel=8   # evaluate across 8 worker processes
```

### See it in the browser

```bash
# terminal 1 - stream the run to disk
php bin/rotifer run flappy_bird --web

# terminal 2 - serve the live dashboard, then open http://localhost:8080
php bin/rotifer serve flappy_bird
```

The web dashboard shows a live fitness chart, the champion's network graph (excitatory/inhibitory
connections, weights on hover, changed wiring flashing before it settles), and an island map
with mutation and trauma levels. From the control panel you can pick a problem, tune it, toggle each
biology mechanism (every option has a hover description), and start/stop runs. When a run finishes the
**champion predictions** table reports a success rate, you can **feed the champion a custom input** and
watch each neuron light up by how strongly it fires, and you can **build a brand-new problem** ("+ New
problem") just by typing in example inputs and the outputs you expect - the engine adapts to it with
defaults recommended for that data.

## Built-in problems

| Name | Kind | Shows off |
|---|---|---|
| `xor` | logic | evolving topology from scratch |
| `memory_recall` | sequence | recurrent memory networks |
| `phone_recall` | memory | recall a phone number from a constant input - pure recurrence |
| `auto_encoder` | unsupervised | compression through a bottleneck |
| `house_price` | regression | ordinary tabular data |
| `weather_forecast` | classification | multi-class output + islands/migration |
| `flappy_bird` | game | emergent control, no training data |

## Teaching it your own task

A new task is **one class**. Define the data, the fitness, and the tuning - that's the entire surface.

```php
namespace Rotifer\Problems;

use Rotifer\Network\Activation\Sigmoid;
use Rotifer\Network\Shape;
use Rotifer\Organism\Organism;
use Rotifer\Runtime\EvolutionConfig;
use Rotifer\Runtime\Fitness\Problem;

final class XorProblem implements Problem
{
    public function name(): string { return 'xor'; }

    public function shape(): Shape { return new Shape(inputs: 3, outputs: 1); }

    public function data(): array
    {
        return [
            [[1, 0, 0], [0]],
            [[1, 0, 1], [1]],
            [[1, 1, 0], [1]],
            [[1, 1, 1], [0]],
        ];
    }

    public function fitness(Organism $organism, array $row): float
    {
        return 1.0 - abs($organism->outputs()[0] - $row[1][0]);
    }

    public function config(): EvolutionConfig
    {
        return EvolutionConfig::default()
            ->population(150)->islands(2)->generations(80)
            ->activation(new Sigmoid())
            ->mutation(weight: 0.85, addNeuron: 0.05, addConnection: 0.12)
            ->adaptiveMutation(true)
            ->migration(everyGenerations: 8, topK: 2)
            ->seed(1234);
    }
}
```

Drop it in `problems/`, then `php bin/rotifer run xor`. A row of `[]` in `data()` resets network memory between sequences. For episodic tasks (games), run the whole episode inside `fitness()` - see `problems/FlappyBirdProblem.php`.

## Testing

```bash
composer test                       # all suites
vendor/bin/phpunit --testsuite Unit
```

Because runs are reproducible, evolution itself is unit-tested (same seed ⇒ identical champion), alongside each genetic and biological mechanism.

> On Windows, run the suite from PowerShell - the parallel tests spawn `php.exe` workers.

## Project layout

```
src/
  Genome/        NodeType, NodeRef, Gene, Genome (+distance), Weight
  Network/       Brain (forward pass), GenomePruner, Activation/, Shape, NetworkSpec
  Organism/      Organism, Epigenome
  Evolution/     World, Island, OrganismFactory, IdSequence,
                 Reproduction/ Selection/
                 Adaptation/ Epigenetics/ Learning/ Migration/
  Runtime/       EvolutionConfig, Rng, Fitness/ (Problem, evaluators, Scorer), Parallel/
  Observe/       EventDispatcher, Event/, Reporter/ (terminal + JSON-stream)
  Persistence/   Codec/ (Json, Binary, Hex), SnapshotStore
  Web/           server.php + public/ (vanilla-JS dashboard)
  Cli/           Console, ProblemRegistry
problems/        one class per task
bin/rotifer      the command-line entry point
```

The original (pre-2.0) implementation is preserved in git history under the
`v1.0.0`-`v1.1.0` tags.

## Requirements

- PHP ≥ 8.2
- Composer
- `amphp/parallel` (pulled in automatically) for `--parallel`

## License

Apache-2.0
