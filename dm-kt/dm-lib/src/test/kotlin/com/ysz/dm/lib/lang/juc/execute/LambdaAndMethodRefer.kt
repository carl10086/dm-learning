package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.Callable
import java.util.concurrent.Executors

/**
 * @author carl
 * @since 2023-01-20 12:18 AM
 **/
class NotRunnable {
    fun go(): Unit = println("NotRunnable")
}

class NotCallable {
    fun get(): Int {
        println("NotCallable")
        return 1
    }
}

fun main(args: Array<String>) {
    val exec = Executors.newCachedThreadPool()

    exec.submit {
        println("Lambda1")
    }

    exec.submit {
        NotRunnable().go()
    }

    exec.submit(Callable {
        println("Lambda2")
        1
    })

    exec.submit(NotCallable()::get)

    exec.shutdown()
}