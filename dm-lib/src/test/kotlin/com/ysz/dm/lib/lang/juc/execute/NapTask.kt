package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.TimeUnit

/**
 *@author carl.yu
 *@since 2023/1/10
 **/
class NapTask(private val id: Int) : Runnable {
    override fun run() {
        /*sleep ?*/
        Nap(0.1)
        println(Thread.currentThread().name)
        /**/
    }

    override fun toString(): String {
        return "NapTask(id=$id)"
    }


}

data class Nap(val t: Double, var msg: String? = null) {
    init {
        try {
            TimeUnit.MILLISECONDS.sleep((1000.0 * t).toLong())
        } catch (e: InterruptedException) {
            /*Java 早期设计的产物, 通过立刻跳出任务来终止它们, 这非常容易产生非常不稳定的状态, 后续欧度不鼓励这样去终止一个任务*/
            throw RuntimeException(e)
        }

        msg?.let {
            println(msg)
        }
    }


}