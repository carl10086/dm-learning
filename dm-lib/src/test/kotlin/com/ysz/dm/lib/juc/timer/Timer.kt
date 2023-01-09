package com.ysz.dm.lib.juc.timer

import java.util.concurrent.TimeUnit


/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/10
 **/
class Timer(
    private val start: Long = System.nanoTime()
) {

    fun duration() = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start)


}