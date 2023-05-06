<?php

namespace GeneticAutoml\Activations;

class Activation
{
    public static function sigmoid(float $value): float
    {
        return 1 / (1 + pow(M_E, -$value));
    }

    public function relu()
    {

    }

    public function tanh()
    {

    }

    public function threshold()
    {

    }
}