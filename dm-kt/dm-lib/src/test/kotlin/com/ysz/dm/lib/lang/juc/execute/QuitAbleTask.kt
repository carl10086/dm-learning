package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * @author carl
 * @since 2023-01-20 12:27 AM
 **/
class QuitAbleTask(private val id: Int) : Runnable {
    private val running = AtomicBoolean(true)

    fun quit(): Unit = running.set(false)

    override fun run() {
        while (running.get()) {
            Nap(0.1)
        }
        println("$id")
    }
}

fun main(args: Array<String>) {
    val count = 150

    val exec = Executors.newCachedThreadPool()

    val tasks = (1..count).map(::QuitAbleTask).onEach { exec.execute(it) }

    Nap(1.0)

    tasks.forEach { it.quit() }
    exec.shutdown()
}

