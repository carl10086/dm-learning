package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * @author carl
 * @since 2023-01-20 12:07 AM
 **/

class CountingTask(val id: Int) : Callable<Int> {
    override fun call(): Int {
        var value = 0

        for (i in 0 until 100) value++

        println("$id ${Thread.currentThread().name} $value")
        return value
    }
}

fun main(args: Array<String>) {
    val exec = Executors.newCachedThreadPool()

    // invoke all tasks nearly at same time
    val futures = exec.invokeAll((0 until 10).map(::CountingTask))

    // 人们不用鼓励使用他 ... CompletableFuture
    val sum = futures.sumOf { it.get() }
    println("sum:$sum")

    exec.shutdown()
}