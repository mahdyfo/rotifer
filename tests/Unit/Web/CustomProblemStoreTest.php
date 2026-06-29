<?php

declare(strict_types=1);

namespace Rotifer\Tests\Unit\Web;

use PHPUnit\Framework\TestCase;
use Rotifer\Web\CustomProblemStore;

final class CustomProblemStoreTest extends TestCase
{
    private string $root;

    protected function setUp(): void
    {
        $this->root = sys_get_temp_dir() . '/rotifer_custom_' . uniqid();
        mkdir($this->root, 0777, true);
    }

    protected function tearDown(): void
    {
        $dir = $this->root . '/problems/custom';
        foreach (glob($dir . '/*.json') ?: [] as $file) {
            unlink($file);
        }
        @rmdir($dir);
        @rmdir($this->root . '/problems');
        @rmdir($this->root);
    }

    public function testSavesValidDefinitionWithCustomPrefix(): void
    {
        $store = new CustomProblemStore($this->root);
        $result = $store->save([
            'name' => 'My Task!',
            'inputs' => 2,
            'outputs' => 1,
            'rows' => [
                ['input' => [0, 0], 'output' => [0]],
                ['input' => [1, 1], 'output' => [1]],
            ],
        ]);

        $this->assertTrue($result['ok']);
        $this->assertSame('custom_my_task', $result['name']);

        $definition = $store->definition('custom_my_task');
        $this->assertSame(2, $definition['inputs']);
        $this->assertCount(2, $definition['rows']);
        $this->assertCount(1, $store->all());
    }

    public function testRejectsRowWithWrongWidth(): void
    {
        $store = new CustomProblemStore($this->root);
        $result = $store->save([
            'name' => 'bad',
            'inputs' => 2,
            'outputs' => 1,
            'rows' => [['input' => [0], 'output' => [0]]], // only 1 input, needs 2
        ]);

        $this->assertFalse($result['ok']);
        $this->assertStringContainsString('input', $result['error']);
    }

    public function testRejectsEmptyRows(): void
    {
        $store = new CustomProblemStore($this->root);
        $result = $store->save(['name' => 'empty', 'inputs' => 1, 'outputs' => 1, 'rows' => []]);

        $this->assertFalse($result['ok']);
    }

    public function testDeleteRemovesDefinition(): void
    {
        $store = new CustomProblemStore($this->root);
        $store->save([
            'name' => 'gone',
            'inputs' => 1,
            'outputs' => 1,
            'rows' => [['input' => [1], 'output' => [1]]],
        ]);

        $this->assertTrue($store->delete('custom_gone'));
        $this->assertNull($store->definition('custom_gone'));
    }
}
