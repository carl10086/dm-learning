package com.ysz.dm.lib.lang.juc.execute

import java.util.concurrent.TimeUnit

/**
 *@author carl.yu
 *@since 2023/1/10
 **/
class NapTask(val id: Int) : Runnable {
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
        } catch (e: Exception) {
            throw RuntimeException(e)
        }

        msg?.let {
            println(msg)
        }
    }


}