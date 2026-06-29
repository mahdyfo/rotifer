<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Persistence;

use PHPUnit\Framework\TestCase;
use Rotifer\Genome\Gene;
use Rotifer\Genome\Genome;
use Rotifer\Genome\NodeType;
use Rotifer\Persistence\Codec\BinaryCodec;
use Rotifer\Persistence\Codec\Codec;
use Rotifer\Persistence\Codec\HexCodec;
use Rotifer\Persistence\Codec\JsonCodec;

final class CodecTest extends TestCase
{
    /** @return array<string, array{Codec}> */
    public static function codecs(): array
    {
        return [
            'json' => [new JsonCodec()],
            'binary' => [new BinaryCodec()],
            'hex' => [new HexCodec()],
        ];
    }

    private function sampleGenome(): Genome
    {
        return new Genome([
            Gene::of(NodeType::Input, 0, NodeType::Hidden, 0, 1.5),
            Gene::of(NodeType::Input, 1, NodeType::Hidden, 0, -2.345678),
            Gene::of(NodeType::Hidden, 0, NodeType::Output, 0, 8.388607),
            Gene::of(NodeType::Input, 65535, NodeType::Output, 3, -0.000001),
        ]);
    }

    /** @dataProvider codecs */
    public function testGenomeRoundTrip(Codec $codec): void
    {
        $genome = $this->sampleGenome();
        $restored = $codec->decode($codec->encode($genome));

        $this->assertSame($genome->count(), $restored->count());
        foreach ($genome->genes() as $i => $gene) {
            $r = $restored->genes()[$i];
            $this->assertSame($gene->connectionKey(), $r->connectionKey());
            // weights survive within the codec's 1e-6 quantization
            $this->assertEqualsWithDelta($gene->weight, $r->weight, 1e-6);
        }
    }

    /** @dataProvider codecs */
    public function testSingleGeneRoundTrip(Codec $codec): void
    {
        $gene = Gene::of(NodeType::Hidden, 42, NodeType::Output, 7, -3.14159);
        $restored = $codec->decodeGene($codec->encodeGene($gene));

        $this->assertSame($gene->connectionKey(), $restored->connectionKey());
        $this->assertEqualsWithDelta($gene->weight, $restored->weight, 1e-6);
    }

    public function testEmptyGenomeRoundTrips(): void
    {
        foreach ([new JsonCodec(), new BinaryCodec(), new HexCodec()] as $codec) {
            $this->assertSame(0, $codec->decode($codec->encode(new Genome()))->count());
        }
    }

    public function testHexIsAsciiSafeAndFixedWidth(): void
    {
        $encoded = (new HexCodec())->encode($this->sampleGenome());
        $this->assertMatchesRegularExpression('/^[0-9a-f]*$/', $encoded);
        $this->assertSame(0, strlen($encoded) % (BinaryCodec::RECORD_BYTES * 2));
    }
}
