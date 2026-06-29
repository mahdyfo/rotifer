<?php

declare(strict_types=1);

namespace Rotifer\Genome;

/**
 * The role a neuron plays in the network.
 *
 * Backed by the same integers the legacy engine used (0/1/2) so genome codecs
 * stay byte-compatible and the values are stable across serialization.
 */
enum NodeType: int
{
    case Input = 0;
    case Hidden = 1;
    case Output = 2;
}
