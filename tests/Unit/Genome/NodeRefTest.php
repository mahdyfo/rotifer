<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Genome;

use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use Rotifer\Genome\NodeRef;
use Rotifer\Genome\NodeType;

final class NodeRefTest extends TestCase
{
    public function testKeyEncodesTypeAndIndex(): void
    {
        $ref = new NodeRef(NodeType::Hidden, 7);
        $this->assertSame('1:7', $ref->key());
    }

    public function testEqualityIsStructural(): void
    {
        $this->assertTrue(NodeRef::input(3)->equals(new NodeRef(NodeType::Input, 3)));
        $this->assertFalse(NodeRef::input(3)->equals(NodeRef::hidden(3)));
        $this->assertFalse(NodeRef::input(3)->equals(NodeRef::input(4)));
    }

    public function testRejectsIndexAboveSixteenBits(): void
    {
        $this->expectException(InvalidArgumentException::class);
        new NodeRef(NodeType::Input, 65536);
    }

    public function testRejectsNegativeIndex(): void
    {
        $this->expectException(InvalidArgumentException::class);
        new NodeRef(NodeType::Input, -1);
    }
}
