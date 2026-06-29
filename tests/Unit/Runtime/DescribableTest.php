<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Runtime;

use PHPUnit\Framework\TestCase;
use Rotifer\Cli\ProblemRegistry;
use Rotifer\Runtime\Fitness\CustomProblem;
use Rotifer\Runtime\Fitness\Describable;

final class DescribableTest extends TestCase
{
    public function testEveryBuiltInProblemHasAShortDescription(): void
    {
        $registry = new ProblemRegistry(dirname(__DIR__, 3) . '/problems');

        $builtIns = array_filter($registry->all(), static fn (array $p): bool => !$p['custom']);
        $this->assertNotEmpty($builtIns, 'expected built-in problems under problems/');

        foreach ($builtIns as $entry) {
            $problem = $registry->resolve($entry['name']);
            $this->assertInstanceOf(Describable::class, $problem, "{$entry['name']} should be Describable");
            $text = $problem->description();
            $this->assertNotSame('', trim($text), "{$entry['name']} should have a non-empty description");
            $this->assertLessThanOrEqual(160, mb_strlen($text), "{$entry['name']} description should stay short");
        }
    }

    public function testCustomProblemUsesItsStoredDescription(): void
    {
        $problem = new CustomProblem([
            'name' => 'custom_demo',
            'inputs' => 2,
            'outputs' => 1,
            'description' => 'My hand-written summary.',
            'rows' => [['input' => [0, 1], 'output' => [1]]],
        ]);

        $this->assertInstanceOf(Describable::class, $problem);
        $this->assertSame('My hand-written summary.', $problem->description());
    }

    public function testCustomProblemFallsBackToAGeneratedDescription(): void
    {
        $problem = new CustomProblem([
            'name' => 'custom_demo',
            'inputs' => 3,
            'outputs' => 2,
            'memory' => true,
            'rows' => [['input' => [0, 0, 1], 'output' => [1, 0]]],
        ]);

        $text = $problem->description();
        $this->assertNotSame('', trim($text));
        $this->assertStringContainsString('3', $text);
        $this->assertStringContainsString('2', $text);
    }
}
