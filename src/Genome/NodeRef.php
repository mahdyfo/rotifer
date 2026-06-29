<?php

declare(strict_types=1);

namespace Rotifer\Genome;

use InvalidArgumentException;

/**
 * An immutable reference to one neuron: its role plus its index within that role.
 *
 * Indexes are bounded to 16 bits (0..65535) because the binary/hex codecs encode
 * each index in two bytes. This is a pure value object - identity is structural.
 */
final readonly class NodeRef
{
    public const MAX_INDEX = 65535;

    public function __construct(
        public NodeType $type,
        public int $index,
    ) {
        if ($index < 0 || $index > self::MAX_INDEX) {
            throw new InvalidArgumentException(
                "Neuron index {$index} is out of the allowed range 0.." . self::MAX_INDEX
            );
        }
    }

    public static function input(int $index): self
    {
        return new self(NodeType::Input, $index);
    }

    public static function hidden(int $index): self
    {
        return new self(NodeType::Hidden, $index);
    }

    public static function output(int $index): self
    {
        return new self(NodeType::Output, $index);
    }

    /** A stable string key, e.g. "1:7", usable as an array key. */
    public function key(): string
    {
        return $this->type->value . ':' . $this->index;
    }

    /** Inverse of {@see key()}. */
    public static function fromKey(string $key): self
    {
        [$type, $index] = explode(':', $key, 2);
        return new self(NodeType::from((int) $type), (int) $index);
    }

    public function equals(self $other): bool
    {
        return $this->type === $other->type && $this->index === $other->index;
    }
}
