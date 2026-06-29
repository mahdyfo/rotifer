<?php

declare(strict_types=1);

namespace Rotifer\Organism;

/**
 * Heritable, decaying "epigenetic" markers - the substrate for genetic trauma.
 *
 * A marker is a named intensity in [0, 1]. Harsh conditions raise it; every
 * generation it decays; at reproduction a child inherits the (decayed) blend of
 * its parents' markers. Active markers don't change the genome - they bias how
 * strongly the next generation mutates - so the effect of an ancestor's stress
 * fades over a few generations, like inherited trauma.
 */
final class Epigenome
{
    /** @param array<string, float> $markers name => intensity */
    public function __construct(private array $markers = [])
    {
    }

    public function raise(string $marker, float $amount): void
    {
        $current = $this->markers[$marker] ?? 0.0;
        $this->markers[$marker] = min(1.0, max(0.0, $current + $amount));
    }

    public function intensity(string $marker = 'stress'): float
    {
        return $this->markers[$marker] ?? 0.0;
    }

    public function total(): float
    {
        return array_sum($this->markers);
    }

    public function isEmpty(): bool
    {
        return $this->total() <= 0.0;
    }

    /** Fade all markers, dropping ones that become negligible. */
    public function decay(float $surviveFactor): void
    {
        foreach ($this->markers as $name => $intensity) {
            $faded = $intensity * $surviveFactor;
            if ($faded < 0.01) {
                unset($this->markers[$name]);
            } else {
                $this->markers[$name] = $faded;
            }
        }
    }

    /** A child's markers: the average of both parents' markers. */
    public static function inherit(self $a, self $b): self
    {
        $names = array_unique([...array_keys($a->markers), ...array_keys($b->markers)]);
        $child = [];
        foreach ($names as $name) {
            $child[$name] = (($a->markers[$name] ?? 0.0) + ($b->markers[$name] ?? 0.0)) / 2.0;
        }
        return new self($child);
    }

    /** @return array<string, float> */
    public function markers(): array
    {
        return $this->markers;
    }
}
