<?php

declare(strict_types=1);

namespace Rotifer\Tests\Support;

use Rotifer\Network\Activation\Activation;

/** Pass-through activation so forward-pass arithmetic is hand-verifiable in tests. */
final class Identity implements Activation
{
    public function activate(float $x): float
    {
        return $x;
    }

    public function name(): string
    {
        return 'identity';
    }
}
