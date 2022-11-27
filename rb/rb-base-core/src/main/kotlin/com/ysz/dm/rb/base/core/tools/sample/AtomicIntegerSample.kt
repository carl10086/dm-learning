package com.ysz.dm.rb.base.core.tools.sample

import java.util.concurrent.atomic.AtomicInteger

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/25
 **/
class AtomicIntegerSample {

    companion object {
        val counter = AtomicInteger(Int.MAX_VALUE - 5)

        fun sample(base: Int): Boolean {
            val count = counter.updateAndGet { if (it < Int.MAX_VALUE) it + 1 else 0 }
            println("count:$count")
            return count % base == 0
        }
    }

}