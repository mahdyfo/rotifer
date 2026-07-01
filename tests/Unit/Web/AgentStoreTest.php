<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Web;

use PHPUnit\Framework\TestCase;
use Rotifer\Web\AgentStore;

final class AgentStoreTest extends TestCase
{
    private string $root;

    protected function setUp(): void
    {
        $this->root = sys_get_temp_dir() . '/rotifer_agents_' . uniqid();
        mkdir($this->root, 0777, true);
    }

    protected function tearDown(): void
    {
        $dir = $this->root . '/saved_agents';
        foreach (glob($dir . '/*.json') ?: [] as $file) {
            unlink($file);
        }
        @rmdir($dir);
        @rmdir($this->root);
    }

    private const GENOME = '000000020000c000'; // opaque hex, as HexCodec would produce

    public function testSavesAndReadsBackAnAgent(): void
    {
        $store = new AgentStore($this->root);
        $result = $store->save([
            'name' => 'My Champ!',
            'problem' => 'xor',
            'inputs' => 2,
            'outputs' => 1,
            'activation' => 'sigmoid',
            'memory' => false,
            'fitness' => 3.9,
            'matchRate' => 0.95,
            'hidden' => 2,
            'geneCount' => 5,
            'genome' => self::GENOME,
        ]);

        $this->assertTrue($result['ok']);
        $this->assertSame('my_champ', $result['name']);
        $this->assertSame('saved_agents/my_champ.json', $result['path']);

        // The listing is a summary: metadata + path, but no genome payload.
        $all = $store->all();
        $this->assertCount(1, $all);
        $this->assertSame('xor', $all[0]['problem']);
        $this->assertSame(5, $all[0]['geneCount']);
        $this->assertSame('saved_agents/my_champ.json', $all[0]['path']);
        $this->assertArrayNotHasKey('genome', $all[0]);

        // The full record keeps the (hex) genome so it can be run.
        $full = $store->get('my_champ');
        $this->assertSame(self::GENOME, $full['genome']);
        $this->assertSame(2, $full['inputs']);
    }

    public function testCompactJsonHasNoWhitespace(): void
    {
        $store = new AgentStore($this->root);
        $store->save(['name' => 'tight', 'genome' => self::GENOME]);
        $raw = file_get_contents($this->root . '/saved_agents/tight.json');
        $this->assertStringNotContainsString("\n", $raw);
        $this->assertStringNotContainsString('": ', $raw); // no pretty-print spacing
    }

    public function testRejectsSaveWithoutGenome(): void
    {
        $store = new AgentStore($this->root);
        $this->assertFalse($store->save(['name' => 'empty'])['ok']);
        $this->assertFalse($store->save(['name' => 'blank', 'genome' => ''])['ok']);
    }

    public function testReusingNameOverwritesTheSlot(): void
    {
        $store = new AgentStore($this->root);
        $store->save(['name' => 'run1', 'genome' => self::GENOME, 'fitness' => 1.0]);
        $store->save(['name' => 'run1', 'genome' => self::GENOME, 'fitness' => 2.0]);

        $this->assertCount(1, $store->all());
        $this->assertSame(2.0, (float) $store->get('run1')['fitness']);
    }

    public function testDeleteRemovesAgent(): void
    {
        $store = new AgentStore($this->root);
        $store->save(['name' => 'gone', 'genome' => self::GENOME]);
        $this->assertTrue($store->delete('gone'));
        $this->assertNull($store->get('gone'));
    }
}
